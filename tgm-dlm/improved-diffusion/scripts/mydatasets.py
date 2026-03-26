from torch.utils.data import DataLoader,Dataset
import torch
import random
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DistributedSampler
RDLogger.DisableLog('rdApp.*')


def get_dataloader(
    dataset,
    batchsize,
    rank,
    world_size,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
):
    sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=True)
    def collate(batch):
        toked_smis= [i['tok_smiles'] for i in batch]
        desc_states = [i['desc_state'] for i in batch]
        desc_mask = [i['desc_mask'] for i in batch]
        corrupted_toked_smis = [i['corrupted_toked_smis'] for i in batch]
        return torch.concat(toked_smis,dim=0),torch.concat(desc_states,dim=0),torch.concat(desc_mask,dim=0),torch.concat(corrupted_toked_smis,dim=0)


    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs,
    )
    def cycle():
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for i in dataloader:
                yield i
            ec+=1 
    return iter(cycle())


def get_latent_dataloader(
    dataset,
    batchsize,
    rank,
    world_size,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
):
    sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=True)

    def collate(batch):
        latents = [i['latent'] for i in batch]
        desc_states = [i['desc_state'] for i in batch]
        desc_masks = [i['desc_mask'] for i in batch]
        return torch.stack(latents,dim=0),torch.concat(desc_states,dim=0),torch.concat(desc_masks,dim=0)

    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor
        dataloader_kwargs["persistent_workers"] = persistent_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs,
    )

    def cycle():
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for i in dataloader:
                yield i
            ec+=1
    return iter(cycle())

class ChEBIdataset(Dataset):
    def __init__(self,dir,smi_tokenizer,split,replace_desc=False,pre=None,prob=0,load_state=True,corrupt_prob=0.4,mask_desc=False):
        super().__init__()
        self.dir = dir
        self.smi_tokenizer = smi_tokenizer
        self.split = split
        self.replace_desc = replace_desc
        self.pre = pre
        self.prob=prob
        self.corrupt_prob = corrupt_prob
        print('corruption prob is {}'.format(self.corrupt_prob))
        self.mask_desc= mask_desc
        print('mask_desc is {}'.format(self.mask_desc))
        import os.path as osp
        split_file = osp.join(self.dir, self.split + '.txt')
        assert osp.exists(split_file), f"split file not found: {split_file}"
        self.ori_data = self.get_ori_data()
        self.load_state=load_state
        if load_state:
            self.desc_state = self.get_desc_state()
    def get_desc_state(self):
        import os.path as osp
        file_path = osp.join(self.dir,self.split+'_desc_states.pt')
        return torch.load(file_path)
    def get_ori_data(self):
        import os.path as osp
        if self.replace_desc:
            import spacy
            nlp = spacy.load('en_core_web_sm')
        res = []
        file_path = osp.join(self.dir,self.split+'.txt')
        with open(file_path,'r') as f:
            for i,line in enumerate(f):
                if i==0: continue
                line = line.split('\t')
                assert len(line)==3
                if line[1]!='*':
                    desc = line[2].strip()
                    if self.replace_desc:
                        doc = nlp(desc)
                        for token in doc:
                            if token.text == 'is':
                                desc = 'The molecule ' + desc[token.idx:]
                                break
                    res.append(
                        (int(line[0]),line[1].strip(),desc)
                    )
        return res
    def __len__(self):
        return len(self.ori_data)
    def permute(self,smiles):
        p = random.random()
        if p<self.prob:
            print("PERMUTE SMILE")
            return changeorder(smiles,shuffle=True)
        else:
            return smiles

    def __getitem__(self,idx):
        data = self.ori_data[idx]
        dic = {'cid':data[0],'smiles':self.permute(data[1]),'desc':data[2]}
        dic['tok_smiles'] = self.smi_tokenizer(dic['smiles'])
        dic['corrupted_toked_smis'] =  self.smi_tokenizer.corrupt(dic['smiles']) if random.random()<self.corrupt_prob else dic['tok_smiles']
        dic['tok_desc'] = None
        dic['dec_mask'] = None
        if self.load_state:
            dic['desc_state'] = self.desc_state[data[0]]['states']
            dic['desc_mask'] = self.desc_state[data[0]]['mask']
            if self.mask_desc:
                dic['desc_state'] = torch.zeros_like(dic['desc_state'])
                dic['desc_mask'] = torch.ones_like(dic['desc_mask'])
        return dic


class LatentChEBIDataset(Dataset):
    def __init__(self,dir,split,latent_file=None,mask_desc=False):
        super().__init__()
        self.dir = dir
        self.split = split
        self.mask_desc = mask_desc
        import os.path as osp
        split_file = osp.join(self.dir, self.split + '.txt')
        assert osp.exists(split_file), f"split file not found: {split_file}"
        self.desc_state = self.get_desc_state()
        self.latents = self.get_latents(latent_file)
        self.ori_data = self.get_ori_data()
        self.ori_data = [
            row for row in self.ori_data
            if row[0] in self.desc_state and row[0] in self.latents
        ]
        print(
            f"latent dataset {self.split}: "
            f"{len(self.ori_data)} usable rows "
            f"(desc={len(self.desc_state)}, latent={len(self.latents)})"
        )

    def get_desc_state(self):
        import os.path as osp
        file_path = osp.join(self.dir,self.split+'_desc_states.pt')
        return torch.load(file_path)

    def get_latents(self,latent_file):
        import os.path as osp
        file_path = latent_file if latent_file is not None else osp.join(self.dir,self.split+'_sdvae_latents.pt')
        return torch.load(file_path)

    def get_ori_data(self):
        import os.path as osp
        res = []
        file_path = osp.join(self.dir,self.split+'.txt')
        with open(file_path,'r') as f:
            for i,line in enumerate(f):
                if i==0:
                    continue
                line = line.split('\t')
                assert len(line)==3
                if line[1] != '*':
                    res.append((int(line[0]),line[1].strip(),line[2].strip()))
        return res

    def __len__(self):
        return len(self.ori_data)

    def __getitem__(self,idx):
        cid, smiles, desc = self.ori_data[idx]
        latent = self.latents[cid]
        if not isinstance(latent, torch.Tensor):
            latent = torch.tensor(latent)
        latent = latent.float().view(-1)
        desc_state = self.desc_state[cid]['states']
        desc_mask = self.desc_state[cid]['mask']
        if self.mask_desc:
            desc_state = torch.zeros_like(desc_state)
            desc_mask = torch.ones_like(desc_mask)
        return {
            'cid':cid,
            'smiles':smiles,
            'desc':desc,
            'latent':latent,
            'desc_state':desc_state,
            'desc_mask':desc_mask,
        }

def changeorder(smiles,shuffle):
    original_smiles = smiles # Replace with your original SMILES string
    # Convert the original SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(original_smiles)
    if mol is None:
        print("Wrong in original dataset")
    Chem.Kekulize(mol)
    # Get the atom indices in the molecule
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    # Reverse the order of the atom indices
    # print(atom_indices)
    # # random.shuffle(atom_indices)
    # # Create a new molecule with the reordered atoms
    # print(atom_indices)
    if shuffle:
        random.shuffle(atom_indices)
    reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
    # if k:
    #     print(reordered_mol)
    # Generate the new SMILES string
    new_smiles = Chem.MolToSmiles(reordered_mol,kekuleSmiles=True)
    return new_smiles
