# explainability/gradcam.py

import torch
import torch.nn.functional as F
import numpy as np


class GradCAM3D:
    """
    3D GradCAM for volumetric CNN models.
    Hooks into a target layer and computes gradient-weighted activation maps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        Generate GradCAM heatmap for a given input tensor.
        Args:
            input_tensor: torch.Tensor of shape (1, 1, 64, 64, 64)
        Returns:
            cam: np.ndarray of shape (64, 64, 64), values in [0, 1]
            det_prob: float, detection probability
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        det_out, mal_out = self.model(input_tensor)
        det_prob = torch.sigmoid(det_out)

        # Backward pass on detection score
        self.model.zero_grad()
        det_out.backward()

        # Compute GradCAM weights via global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, det_prob.item()


def visualize_gradcam(ct_cube, cam, detection_prob, series_uid="", nodule_idx=0, save_dir="."):
    """
    Plot and save a 6-panel GradCAM visualization.
    Top row: raw CT slices. Bottom row: GradCAM overlaid on CT.
    Args:
        ct_cube       : np.ndarray (64,64,64) normalized HU
        cam           : np.ndarray (64,64,64) GradCAM heatmap
        detection_prob: float
        series_uid    : str
        nodule_idx    : int
        save_dir      : str, folder to save PNG
    Returns:
        save_path: str
    """
    import matplotlib.pyplot as plt
    import os

    center = ct_cube.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('#0f0f0f')

    slices = {
        'Axial (Z)':    (ct_cube[center, :, :],  cam[center, :, :]),
        'Coronal (Y)':  (ct_cube[:, center, :],  cam[:, center, :]),
        'Sagittal (X)': (ct_cube[:, :, center],  cam[:, :, center]),
    }

    for col, (title, (ct_slice, cam_slice)) in enumerate(slices.items()):
        # Row 0 - CT only
        axes[0, col].imshow(ct_slice, cmap='gray', interpolation='bilinear')
        axes[0, col].set_title(f'{title}\nCT Scan', color='white', fontsize=11)
        axes[0, col].axis('off')

        # Row 1 - CT + GradCAM overlay
        axes[1, col].imshow(ct_slice, cmap='gray', interpolation='bilinear')
        axes[1, col].imshow(cam_slice, cmap='jet', alpha=0.45, interpolation='bilinear')
        axes[1, col].set_title(f'{title}\nGradCAM Overlay', color='white', fontsize=11)
        axes[1, col].axis('off')

    status = "NODULE DETECTED" if detection_prob > 0.5 else "NOT A NODULE"
    color  = "#00ff88"         if detection_prob > 0.5 else "#ff4444"

    fig.suptitle(
        f'{status}   |   Confidence: {detection_prob:.1%}\n'
        f'Scan: {series_uid[:50]}...',
        color=color, fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'gradcam_nodule_{nodule_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    plt.show()
    plt.close()

    return save_path# explainability/gradcam.py

import torch
import torch.nn.functional as F
import numpy as np


class GradCAM3D:
    """
    3D GradCAM for volumetric CNN models.
    Hooks into a target layer and computes gradient-weighted activation maps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        """
        Generate GradCAM heatmap for a given input tensor.
        Args:
            input_tensor: torch.Tensor of shape (1, 1, 64, 64, 64)
        Returns:
            cam: np.ndarray of shape (64, 64, 64), values in [0, 1]
            det_prob: float, detection probability
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        det_out, mal_out = self.model(input_tensor)
        det_prob = torch.sigmoid(det_out)

        # Backward pass on detection score
        self.model.zero_grad()
        det_out.backward()

        # Compute GradCAM weights via global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, det_prob.item()


def visualize_gradcam(ct_cube, cam, detection_prob, series_uid="", nodule_idx=0, save_dir="."):
    """
    Plot and save a 6-panel GradCAM visualization.
    Top row: raw CT slices. Bottom row: GradCAM overlaid on CT.
    Args:
        ct_cube       : np.ndarray (64,64,64) normalized HU
        cam           : np.ndarray (64,64,64) GradCAM heatmap
        detection_prob: float
        series_uid    : str
        nodule_idx    : int
        save_dir      : str, folder to save PNG
    Returns:
        save_path: str
    """
    import matplotlib.pyplot as plt
    import os

    center = ct_cube.shape[0] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('#0f0f0f')

    slices = {
        'Axial (Z)':    (ct_cube[center, :, :],  cam[center, :, :]),
        'Coronal (Y)':  (ct_cube[:, center, :],  cam[:, center, :]),
        'Sagittal (X)': (ct_cube[:, :, center],  cam[:, :, center]),
    }

    for col, (title, (ct_slice, cam_slice)) in enumerate(slices.items()):
        # Row 0 - CT only
        axes[0, col].imshow(ct_slice, cmap='gray', interpolation='bilinear')
        axes[0, col].set_title(f'{title}\nCT Scan', color='white', fontsize=11)
        axes[0, col].axis('off')

        # Row 1 - CT + GradCAM overlay
        axes[1, col].imshow(ct_slice, cmap='gray', interpolation='bilinear')
        axes[1, col].imshow(cam_slice, cmap='jet', alpha=0.45, interpolation='bilinear')
        axes[1, col].set_title(f'{title}\nGradCAM Overlay', color='white', fontsize=11)
        axes[1, col].axis('off')

    status = "NODULE DETECTED" if detection_prob > 0.5 else "NOT A NODULE"
    color  = "#00ff88"         if detection_prob > 0.5 else "#ff4444"

    fig.suptitle(
        f'{status}   |   Confidence: {detection_prob:.1%}\n'
        f'Scan: {series_uid[:50]}...',
        color=color, fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'gradcam_nodule_{nodule_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    plt.show()
    plt.close()

    return save_path