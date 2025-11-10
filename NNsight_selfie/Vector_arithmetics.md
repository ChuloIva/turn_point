
## **Vector Interpolation (Slerp/Lerp)**

```python
def interpolate_concepts(vec_a, vec_b, alpha=0.5, method="linear"):
    """Smoothly blend between two concepts"""
    if method == "linear":
        return (1 - alpha) * vec_a + alpha * vec_b
    elif method == "spherical":
        # Spherical linear interpolation
        dot = torch.dot(vec_a, vec_b)
        omega = torch.acos(torch.clamp(dot, -1, 1))
        return (torch.sin((1-alpha)*omega) * vec_a + torch.sin(alpha*omega) * vec_b) / torch.sin(omega)

# Example: Blend happy and sad to get neutral
happy_sad_blend = interpolate_concepts(happy_vec, sad_vec, 0.3)  # 30% toward sad

```



## **Vector Rotation/Transformation**

```python
def rotate_in_plane(vector, reference_vec, angle):
    """Rotate vector in the plane defined by vector and reference"""
    # Gram-Schmidt to create orthonormal basis
    u1 = F.normalize(vector.unsqueeze(0), dim=1).squeeze(0)
    u2_unnorm = reference_vec - torch.dot(reference_vec, u1) * u1
    u2 = F.normalize(u2_unnorm.unsqueeze(0), dim=1).squeeze(0)

    # Rotate
    return torch.cos(angle) * u1 + torch.sin(angle) * u2

# Example: Rotate "happy" around "emotional" axis
rotated_happy = rotate_in_plane(happy_vec, emotional_axis, torch.pi/4)

```