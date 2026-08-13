#simple gcn to add graph information to actor critic

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

class SMILES_to_Graph:
    #process smiles to graph and node features
    def __init__(self, max_atoms=100):
        self.max_atoms = max_atoms
        self.atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']  # common atom types
        self.atom_type_to_idx = {atom: idx for idx, atom in enumerate(self.atom_types)}
        self.num_atom_types = len(self.atom_types)

        #bond types: single, double, triple, aromatic
        self.bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.bond_type_to_idx = {bond: idx for idx, bond in enumerate(self.bond_types)}
        self.num_bond_types = len(self.bond_types)
    
    def featurize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_atoms:
            raise ValueError(f"Number of atoms {num_atoms} exceeds max_atoms {self.max_atoms}")

        # Node features
        node_features = torch.zeros((self.max_atoms, self.num_atom_types))
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            atom_type = atom.GetSymbol()
            if atom_type in self.atom_type_to_idx:
                node_features[idx, self.atom_type_to_idx[atom_type]] = 1.0

        # Adjacency matrix
        adj = torch.zeros((self.max_atoms, self.max_atoms))
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            adj[begin_idx, end_idx] = 1.0
            adj[end_idx, begin_idx] = 1.0  # undirected graph

        return node_features, adj

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_features]
        # adj: [batch_size, num_nodes, num_nodes]
        
        batch_size, num_nodes, _ = x.shape
        
        # add self-loops
        eye = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye
        
        # compute degree and normalize
        degree = torch.sum(adj, dim=2)  # [batch_size, num_nodes]
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.
        
        # D^{-1/2} A D^{-1/2}
        # Using batch matrix multiplication
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [batch_size, num_nodes, num_nodes]
        adj_normalized = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)
        
        x = torch.bmm(adj_normalized, x)  # [batch_size, num_nodes, in_features]
        x = self.linear(x)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj):
        for conv in self.convs:
            x = F.relu(conv(x, adj))
        # mean pooling over nodes (dim=1 for batched input)
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        pred= self.out(x)
        return x, pred

class SharedGNNEncoder(nn.Module):
    """
    Shared Siamese-type GNN encoder that computes:
    z_b = gnn(x_before, adj_before, mask_before)
    z_a = gnn(x_after, adj_after, mask_after)
    delta = z_a - z_b
    pred_y = mlp(delta)
    """
    def __init__(self, node_feature_dim, hidden_dim, output_dim,num_layers=2):
        super(SharedGNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(node_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        
        # MLP for predicting from delta
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.hidden_dim = hidden_dim

    def encode(self, x, adj, mask=None):
        """Encode a single graph to get node representations and pooled output."""
        for conv in self.convs:
            x = F.relu(conv(x, adj))
        # mean pooling over nodes (dim=1 for batched input)
        z = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        if mask is not None:
            z = z + mask
        return z

    def forward(self, x_before, adj_before, x_after, adj_after):
        """
        Forward pass computing delta between before and after states.
        
        Args:
            x_before: Node features before action [batch_size, num_nodes, node_feature_dim]
            adj_before: Adjacency matrix before action [batch_size, num_nodes, num_nodes]
            x_after: Node features after action [batch_size, num_nodes, node_feature_dim]
            adj_after: Adjacency matrix after action [batch_size, num_nodes, num_nodes]
            mask_before: Optional mask for before state [batch_size, hidden_dim]
            mask_after: Optional mask for after state [batch_size, hidden_dim]
        
        Returns:
            delta: Difference in representations [batch_size, hidden_dim]
            pred_y: Predicted output from MLP [batch_size, output_dim]
        """
        # Encode before state
        z_b = self.encode(x_before, adj_before)
        
        # Encode after state
        z_a = self.encode(x_after, adj_after)
        
        # Compute delta
        delta = z_a - z_b
        
        # Predict from delta
        pred_y = self.mlp(delta)
        
        return delta, pred_y
    
    def encode_single(self, x, adj, mask=None):
        """Convenience method to encode a single graph (for inference)."""
        return self.encode(x, adj, mask)