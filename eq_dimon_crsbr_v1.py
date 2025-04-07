import numpy as np
import torch
import torch.nn as nn
from torch.func import grad, vmap
import pennylane as qml
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Device and seeds
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Initializing with device: {device}")

# --- CrSBr-Specific Domain and Physics ---
def generate_crbr_domain(theta, N=64, adaptive=False):
    """Generate CrSBr domain with optional adaptive sampling based on strain."""
    strain, mag_dir, phase = theta
    if adaptive:
        # Higher density where strain gradients are steep
        x_base = np.linspace(0, 1.5, N)
        y_base = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x_base, y_base)
        strain_field = strain * np.cos(mag_dir * 2 * np.pi + phase) * (X / 1.5)
        grad_x = np.gradient(strain_field, x_base, axis=1)
        sampling_density = 1 + 5 * np.abs(grad_x) / np.max(np.abs(grad_x) + 1e-8)
        x_adapt = np.linspace(0, 1.5, int(N * np.mean(sampling_density)))
        X, Y = np.meshgrid(x_adapt, y_base)
    else:
        X, Y = np.meshgrid(np.linspace(0, 1.5, N), np.linspace(0, 1, N))
    
    x = X + strain * np.cos(mag_dir * 2 * np.pi + phase) * (X / 1.5) * (1 + 0.1 * np.sin(Y * np.pi))
    y = Y + strain * np.sin(mag_dir * 2 * np.pi + phase)**2 * (Y / 1.0)
    return x, y

def diffeomorphism_crbr(x, y, theta):
    strain, mag_dir, phase = theta
    X = x - strain * np.cos(mag_dir * 2 * np.pi + phase) * (x / 1.5) * (1 + 0.1 * np.sin(y * np.pi))
    Y = y - strain * np.sin(mag_dir * 2 * np.pi + phase)**2 * (y / 1.0)
    return X, Y

def reference_domain(N=64):
    return np.stack(np.meshgrid(np.linspace(0, 1.5, N), np.linspace(0, 1, N)), axis=-1)

def subsampled_reference_domain(N=8):
    return np.stack(np.meshgrid(np.linspace(0, 1.5, N), np.linspace(0, 1, N)), axis=-1)

def solve_poisson_crbr(theta, bc, rho, A_field, N=64):
    """Solve Poisson's equation with magnetic vector potential influence."""
    x, y = generate_crbr_domain(theta, N)
    u = np.zeros((N, N))
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = bc
    epsilon = 10.0  # CrSBr permittivity
    h = 1.5 / (N - 1)
    Ax, Ay = A_field  # Magnetic vector potential components
    for _ in range(5000):
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) - \
                         (h**2) * (rho[1:-1, 1:-1] / (4 * epsilon) + Ax[1:-1, 1:-1] + Ay[1:-1, 1:-1])
    return u

# --- EnhancedMIONet with 6-Qubit Quantum Layer ---
class EnhancedMIONet(nn.Module):
    def __init__(self, theta_dim=3, bc_dim=4, rho_dim=64, hidden_dim=512, num_quantum_weights=18):
        super().__init__()
        self.device = device
        
        self.branch_theta = nn.Sequential(nn.Linear(theta_dim, hidden_dim), nn.Tanh(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.branch_bc = nn.Sequential(nn.Linear(bc_dim, hidden_dim), nn.Tanh(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.branch_rho = nn.Sequential(nn.Linear(rho_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                  nn.Linear(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        # 6-qubit quantum circuit with variational layers
        self.quantum_weights = nn.Parameter(torch.randn(num_quantum_weights, device=device) * 0.1)
        self.qdev = qml.device("default.qubit.torch", wires=6, torch_device="cuda" if torch.cuda.is_available() else "cpu")
        
        @qml.qnode(self.qdev, interface='torch')
        def quantum_circuit(inputs, weights):
            inputs = torch.pi * (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
            # Layer 1: Encode inputs
            for i in range(6):
                qml.RY(inputs[..., i], wires=i)
                qml.RX(weights[i], wires=i)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[4, 5])
            # Layer 2: Variational
            for i in range(6):
                qml.RZ(inputs[..., i + 3], wires=i)
                qml.RX(weights[i + 6], wires=i)
            qml.CZ(wires=[1, 2])
            qml.CZ(wires=[3, 4])
            # Layer 3: Entanglement
            for i in range(6):
                qml.RY(weights[i + 12], wires=i)
            qml.CNOT(wires=[0, 5])
            qml.CZ(wires=[2, 4])
            return [qml.expval(qml.PauliZ(w)) for w in range(6)]
        
        self.quantum_circuit = quantum_circuit
    
    def quantum_layer(self, theta, bc, rho):
        rho_stats = torch.stack([rho.mean(dim=1), rho.std(dim=1), rho.max(dim=1)[0]], dim=-1)  # [batch, 3]
        inputs = torch.cat([theta, bc, rho_stats], dim=-1)  # [batch, 10]
        q_out = torch.stack(self.quantum_circuit(inputs[:, :9], self.quantum_weights), dim=-1)  # [batch, 6]
        return q_out.mean(dim=-1)  # Scalar per batch
    
    def forward(self, theta, bc, rho, X_ref):
        batch_size = theta.shape[0]
        n_points = X_ref.shape[-2]
        
        theta_out = self.branch_theta(theta).unsqueeze(1).expand(-1, n_points, -1)
        bc_out = self.branch_bc(bc).unsqueeze(1).expand(-1, n_points, -1)
        rho_out = self.branch_rho(rho).unsqueeze(1).expand(-1, n_points, -1)
        trunk_out = self.trunk(X_ref)
        if trunk_out.dim() == 2:
            trunk_out = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)
        
        combined = theta_out * bc_out * rho_out * trunk_out
        q_factor = self.quantum_layer(theta, bc, rho).unsqueeze(-1).unsqueeze(-1)
        return self.final_layer(combined) * (1 + q_factor)

# --- eQ-DIMON for CrSBr ---
class eQ_DIMON:
    def __init__(self, batch_size=64, initial_lr=0.001, weight_decay=1e-4, quantum_weight=1.0):
        self.device = device
        self.model = EnhancedMIONet(rho_dim=64).to(device)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.quantum_weight = quantum_weight

    def compute_poisson_residual(self, theta_batch, bc_batch, rho_batch, X_ref_sub):
        try:
            X_ref_tensor = torch.tensor(X_ref_sub.reshape(-1, 2), dtype=torch.float32, device=self.device, requires_grad=True)
            theta_tensor = torch.tensor(theta_batch, dtype=torch.float32, device=self.device)
            bc_tensor = torch.tensor(bc_batch, dtype=torch.float32, device=self.device)
            rho_tensor = torch.tensor(rho_batch, dtype=torch.float32, device=self.device)
            
            u = self.model(theta_tensor, bc_tensor, rho_tensor, X_ref_tensor).squeeze(-1)
            grad_u = torch.autograd.grad(u.sum(), X_ref_tensor, create_graph=True)[0]
            u_x, u_y = grad_u[:, 0].view(self.batch_size, -1), grad_u[:, 1].view(self.batch_size, -1)
            u_xx = torch.autograd.grad(u_x.sum(), X_ref_tensor, create_graph=True)[0][:, 0].view(self.batch_size, -1)
            u_yy = torch.autograd.grad(u_y.sum(), X_ref_tensor, create_graph=True)[0][:, 1].view(self.batch_size, -1)
            
            epsilon = 10.0
            laplacian = u_xx + u_yy
            rho_target = rho_tensor.view(self.batch_size, -1)
            residual = torch.mean((laplacian + rho_target / epsilon)**2)
            return residual
        except Exception as e:
            logging.error(f"PDE computation failed: {str(e)}")
            return torch.tensor(float('inf'), device=self.device)

    def _train_batch(self, batch, X_ref_sub, X_ref_full):
        theta_batch, bc_batch, rho_batch, u_batch = batch
        theta_batch, bc_batch, rho_batch, u_batch = [x.to(self.device) for x in batch]
        u_batch = u_batch.view(-1, 4096)
        X_ref_full_tensor = torch.tensor(X_ref_full.reshape(-1, 2), dtype=torch.float32, device=self.device)
        
        self.optimizer.zero_grad()
        u_pred = self.model(theta_batch, bc_batch, rho_batch, X_ref_full_tensor).squeeze(-1)
        mse_loss = torch.mean((u_pred - u_batch)**2)
        
        pde_loss = self.compute_poisson_residual(theta_batch, bc_batch, rho_batch, X_ref_sub)
        bc_pred = u_pred[:, [0, 63, 4032, 4095]]
        bc_loss = torch.mean((bc_pred - bc_batch)**2)
        
        quantum_params = self.model.quantum_weights
        quantum_penalty = torch.mean((quantum_params - 0.5)**2)
        
        loss = mse_loss + 10.0 * pde_loss + bc_loss + self.quantum_weight * quantum_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), mse_loss.item(), pde_loss.item(), bc_loss.item()

    def train(self, data, epochs=500, patience=20, val_split=0.2):
        theta_data, bc_data, rho_data, u_data = zip(*data)
        theta_data = torch.tensor(np.stack(theta_data), dtype=torch.float32)
        bc_data = torch.tensor(np.stack(bc_data), dtype=torch.float32)
        rho_data = torch.tensor(np.stack(rho_data), dtype=torch.float32)
        u_data = torch.tensor(np.stack(u_data), dtype=torch.float32)
        
        n_samples = len(data)
        n_val = int(val_split * n_samples)
        n_train = n_samples - n_val
        n_train_batches = n_train // self.batch_size
        n_train = n_train_batches * self.batch_size
        
        perm = torch.randperm(n_samples)
        train_idx, val_idx = perm[:n_train], perm[n_train:n_train + n_val]
        
        train_data = [theta_data[train_idx], bc_data[train_idx], rho_data[train_idx], u_data[train_idx]]
        val_data = [theta_data[val_idx], bc_data[val_idx], rho_data[val_idx], u_data[val_idx]]
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for i in range(0, n_train, self.batch_size):
                batch = [x[i:i + self.batch_size] for x in train_data]
                losses = self._train_batch(batch, X_ref_sub, X_ref_full)
                epoch_loss += losses[0]
            train_losses.append(epoch_loss / n_train_batches)
            
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                n_val_batches = len(val_idx) // self.batch_size
                for i in range(0, n_val_batches * self.batch_size, self.batch_size):
                    batch = [x[i:i + self.batch_size] for x in val_data]
                    u_pred = self.model(*[x.to(self.device) for x in batch[:3]], 
                                      torch.tensor(X_ref_full.reshape(-1, 2), device=self.device)).squeeze(-1)
                    val_loss += torch.mean((u_pred - batch[3].to(self.device).view(-1, 4096))**2).item()
                val_loss /= n_val_batches if n_val_batches > 0 else 1
                val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            logging.info(f"Epoch {epoch}: Train Loss={train_losses[-1]:.6f}, Val Loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            elif patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
            else:
                patience_counter += 1
        
        return train_losses, val_losses

    def predict(self, theta, bc, rho, X_ref):
        inputs = [torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0) for x in [theta, bc, rho]]
        X_flat = torch.tensor(X_ref.reshape(-1, 2), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            u_pred = self.model(*inputs, X_flat).squeeze().cpu().numpy().reshape(X_ref.shape[:2])
        u_pred[0, :], u_pred[-1, :], u_pred[:, 0], u_pred[:, -1] = bc
        return u_pred

# --- Data Generation ---
def generate_crbr_data_worker(args):
    theta, bc = args
    X_ref = reference_domain()
    mag_dir = theta[1]
    rho = 0.1 * np.sin(mag_dir * 2 * np.pi * X_ref[..., 0]) * np.cos(X_ref[..., 1] * np.pi)
    Ax = 0.05 * np.cos(mag_dir * 2 * np.pi * X_ref[..., 0])  # Magnetic vector potential
    Ay = 0.05 * np.sin(mag_dir * 2 * np.pi * X_ref[..., 1])
    A_field = (Ax, Ay)
    u = solve_poisson_crbr(theta, bc, rho, A_field)
    X_mapped, Y_mapped = diffeomorphism_crbr(*generate_crbr_domain(theta), theta)
    u_ref = griddata((X_mapped.flatten(), Y_mapped.flatten()), u.flatten(), 
                     (X_ref[..., 0], X_ref[..., 1]), method='cubic')
    return (theta, bc, rho, u_ref)

def generate_crbr_data(n_samples=300):
    pool = mp.Pool(mp.cpu_count())
    thetas = np.random.uniform([0, 0, -0.5], [0.03, 1.0, 0.5], (n_samples, 3))
    bcs = np.random.uniform(-1, 1, (n_samples, 4))
    data = pool.map(generate_crbr_data_worker, zip(thetas, bcs))
    pool.close()
    return data

# --- Main Execution ---
if __name__ == '__main__':
    start_time = time()
    logging.info("Generating CrSBr data with quantum pizzazz...")
    try:
        data = generate_crbr_data(n_samples=300)
    except Exception as e:
        logging.error(f"Data generation failed: {str(e)}")
        exit(1)
    
    logging.info("Training eQ-DIMON for CrSBr - Quantum Edition!")
    eq_dimon = eQ_DIMON(batch_size=64)
    X_ref_sub = subsampled_reference_domain(N=8)
    X_ref_full = reference_domain()
    train_losses, val_losses = eq_dimon.train(data, epochs=10, patience=5)
    
    logging.info("Testing on CrSBr samples with quantum flair...")
    test_indices = np.random.choice(len(data), 5, replace=False)
    test_mses = []
    X_ref = reference_domain()
    
    fig = plt.figure(figsize=(20, 8))
    for i, idx in enumerate(test_indices):
        theta, bc, rho, u_true = data[idx]
        u_pred = eq_dimon.predict(theta, bc, rho, X_ref)
        mse = np.mean((u_pred - u_true)**2)
        test_mses.append(mse)
        logging.info(f"CrSBr Sample {idx}: theta={theta}, bc={bc}, MSE={mse:.6f}")
        
        x_test, y_test = generate_crbr_domain(theta, adaptive=True)
        mag_dir = theta[1]
        Bx = np.sin(mag_dir * 2 * np.pi * x_test)
        By = np.cos(mag_dir * 2 * np.pi * y_test)
        
        ax = fig.add_subplot(2, 5, i + 1)
        ax.contourf(x_test, y_test, u_true, levels=20, cmap='viridis')
        ax.streamplot(x_test, y_test, Bx, By, color='white', linewidth=0.5)
        ax.set_title(f"True (MSE: {mse:.6f})")
        
        ax = fig.add_subplot(2, 5, i + 6)
        u_pred_interp = griddata((X_ref[..., 0].flatten(), X_ref[..., 1].flatten()), 
                                u_pred.flatten(), (x_test, y_test), method='cubic')
        ax.contourf(x_test, y_test, u_pred_interp, levels=20, cmap='viridis')
        ax.streamplot(x_test, y_test, Bx, By, color='white', linewidth=0.5)
        ax.set_title("Predicted")
    
    plt.tight_layout()
    plt.show()
    
    q_weights = eq_dimon.model.quantum_weights.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(q_weights, 'o-', label='Quantum Weights')
    plt.axhline(0.5, color='r', linestyle='--', label='Target Mean')
    plt.title("Quantum Circuit Weights - CrSBr Influence")
    plt.legend()
    plt.show()
    
    logging.info(f"Average Test MSE: {np.mean(test_mses):.6f}")
    logging.info(f"Total runtime: {time() - start_time:.2f} seconds")