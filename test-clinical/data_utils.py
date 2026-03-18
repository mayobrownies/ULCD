import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import re
import os
from . import config

# -----------------
# 1. Label Extraction & Cleaning
# -----------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    return text

def get_label(text):
    # Classes: Home, Home Health, SNF, Rehab, Expired, Other
    # Mapping
    t = clean_text(text)
    
    if re.search(r'\bexpired\b|\bdied\b', t):
        return 4 # Expired
    elif re.search(r'\bhome health\b|\bhome with service\b', t):
        return 1 # Home Health
    elif re.search(r'\bsnf\b|\bskilled nursing\b', t):
        return 2 # SNF
    elif re.search(r'\brehab\b', t):
        return 3 # Rehab
    elif re.search(r'\bhome\b', t):
        return 0 # Home
    else:
        return 5 # Other

LABEL_MAP = {
    0: "Home",
    1: "Home Health",
    2: "SNF",
    3: "Rehab",
    4: "Expired",
    5: "Other"
}

# -----------------
# 2. Dataset Class
# -----------------

class MimicDispositionDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------
# 3. Data Loading & Partitioning
# -----------------

def load_mimic_data():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(config.CACHE_DIR, "mimic_clinical_bert_cache.pt")
    
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = torch.load(cache_path, weights_only=False)
        return data['df'], data['input_ids'], data['attention_masks']

    print(f"Loading MIMIC BHC from {config.MIMIC_PATH}...")
    df_bhc = pd.read_csv(config.MIMIC_PATH)
    
    print(f"Loading Discharge Info from {config.NOTE_DISCHARGE_PATH}...")
    df_discharge = pd.read_csv(config.NOTE_DISCHARGE_PATH, compression='gzip', usecols=['note_id', 'subject_id'])
    
    # Join on note_id
    print("Joining datasets on note_id...")
    df = pd.merge(df_bhc, df_discharge, on='note_id', how='inner')
    
    # Extract Labels
    print("Extracting labels...")
    df['clean_target'] = df['target'].apply(clean_text)
    df['label'] = df['target'].apply(get_label)
    
    print(f"Loaded {len(df)} samples. Class dist: {df['label'].value_counts().to_dict()}")
    
    # Pre-tokenize
    print("Pre-tokenizing dataset in chunks (to ensure stability)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    texts = df['input'].astype(str).tolist()
    
    input_ids_list = []
    attention_masks_list = []
    
    chunk_size = 2000 # Process 2000 docs at a time
    total = len(texts)
    
    for i in range(0, total, chunk_size):
        batch_texts = texts[i : i + chunk_size]
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids_list.append(encodings['input_ids'])
        attention_masks_list.append(encodings['attention_mask'])
        
        if (i + chunk_size) % 10000 < chunk_size:
             print(f"  Processed {min(i + chunk_size, total)}/{total} samples...")
    
    print("Concatenating tensors...")
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    
    # Save cache
    print(f"Saving cache to {cache_path}...")
    torch.save({
        'df': df,
        'input_ids': input_ids,
        'attention_masks': attention_masks
    }, cache_path)
    
    return df, input_ids, attention_masks

def partition_patients(df, num_clients, alpha):
    # Partition SUBJECT_IDs
    patient_labels = df.groupby('subject_id')['label'].agg(lambda x: x.mode()[0])
    unique_patients = patient_labels.index.values
    patient_classes = patient_labels.values
    
    num_classes = config.NUM_CLASSES
    client_patient_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_k = np.where(patient_classes == c)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        ## Balance slightly
        proportions = np.array([p * (len(idx_j) < len(unique_patients) / num_clients) for p, idx_j in zip(proportions, client_patient_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        split_idx = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_patient_indices[i].extend(unique_patients[split_idx[i]])

    return client_patient_indices

def get_heterogeneous_dataloaders():
    # Load DF and pre-tokenized tensors
    df, all_input_ids, all_attention_masks = load_mimic_data()
    
    # Ensure they align
    assert len(df) == len(all_input_ids)
    
    # 1. Split Subjects into Global Train (70%), Val (10%), Test (20%)
    all_subjects = df['subject_id'].unique()
    train_subs, test_subs = train_test_split(all_subjects, test_size=0.2, random_state=config.SEED)
    train_subs, val_subs = train_test_split(train_subs, test_size=0.125, random_state=config.SEED) 
    
    # Create masks/indices for splits
    # It's easier to work with indices into the big tensors
    
    # Map subject_id to indices
    # This is a bit slow. Let's add a split column to df temporarily or just use boolean masking
    
    # Train Indices
    train_mask = df['subject_id'].isin(train_subs).values
    val_mask = df['subject_id'].isin(val_subs).values
    test_mask = df['subject_id'].isin(test_subs).values
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    train_df = df.iloc[train_indices]
    
    print(f"Global Split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    # 2. Partition Train Subjects among Clients
    client_patient_ids = partition_patients(train_df, config.NUM_CLIENTS, config.DIRICHLET_ALPHA)
    
    client_loaders = []
    client_test_loaders = []
    client_classes_list = []
    
    for i, pids in enumerate(client_patient_ids):
        # Find indices for this client
        c_mask = train_df['subject_id'].isin(pids).values
        c_global_indices = train_indices[c_mask]
        
        if len(c_global_indices) == 0:
            print(f"WARNING: Client {i} has no data!")
            client_loaders.append(None)
            client_test_loaders.append(None)
            client_classes_list.append([])
            continue
            
        # Local Classes (from labels)
        c_labels = df.iloc[c_global_indices]['label'].values
        c_unique_labels = sorted(list(set(c_labels)))
        client_classes_list.append(c_unique_labels)
        
        # Local Split
        client_sub_df = df.iloc[c_global_indices]
        c_subs = client_sub_df['subject_id'].unique()
        
        if len(c_subs) > 1:
            c_train_subs, c_test_subs = train_test_split(c_subs, test_size=0.2, random_state=config.SEED)
            c_train_mask = client_sub_df['subject_id'].isin(c_train_subs).values
            c_test_mask = client_sub_df['subject_id'].isin(c_test_subs).values
            
            c_train_global_indices = c_global_indices[c_train_mask]
            c_test_global_indices = c_global_indices[c_test_mask]
        else:
            c_train_global_indices = c_global_indices
            c_test_global_indices = []

        # Create Datasets using slicing
        train_ds = MimicDispositionDataset(
            all_input_ids[c_train_global_indices],
            all_attention_masks[c_train_global_indices],
            df.iloc[c_train_global_indices]['label'].values
        )
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY
        )
        
        test_loader = None
        if len(c_test_global_indices) > 0:
            test_ds = MimicDispositionDataset(
                all_input_ids[c_test_global_indices],
                all_attention_masks[c_test_global_indices],
                df.iloc[c_test_global_indices]['label'].values
            )
            test_loader = DataLoader(
                test_ds, 
                batch_size=config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=config.DATALOADER_NUM_WORKERS
            )
            
        client_loaders.append(train_loader)
        client_test_loaders.append(test_loader)

    # Val Loader
    val_ds = MimicDispositionDataset(
        all_input_ids[val_indices],
        all_attention_masks[val_indices],
        df.iloc[val_indices]['label'].values
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.DATALOADER_NUM_WORKERS)
    
    # Test Loader
    test_ds = MimicDispositionDataset(
        all_input_ids[test_indices],
        all_attention_masks[test_indices],
        df.iloc[test_indices]['label'].values
    )
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.DATALOADER_NUM_WORKERS)
    
    public_loaders = [None] * config.NUM_CLIENTS
    
    client_info = {
         'client_test_loaders': client_test_loaders,
         'classes': client_classes_list,
         'num_classes': config.NUM_CLASSES
    }

    
    return client_loaders, public_loaders, test_loader, client_info
