import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import torch
import torch.nn.functional as F

def dense_srm_refinement(img, prob, sxy_bilateral=65, srgb_bilateral=10, sxy_gaussian=3,
                         compat_bilateral=7, compat_gaussian=5, n_iters=5):
    h, w = prob.shape[1:]
    n_classes = 2

    # Create SRM object (DenseCRF implementation)
    d = dcrf.DenseCRF2D(w, h, n_classes)

    # Set unary energy (negative log probability)
    unary = unary_from_softmax(prob)
    d.setUnaryEnergy(unary)

    # Add bilateral energy (appearance/color term)
    pairwise_energy = create_pairwise_bilateral(
        sdims=(sxy_bilateral, sxy_bilateral),
        schan=(srgb_bilateral,),
        img=img,
        chdim=2
    )
    d.addPairwiseEnergy(pairwise_energy, compat=compat_bilateral)

    # Add Gaussian energy (smoothing term)
    pairwise_energy_gaussian = create_pairwise_gaussian(
        sdims=(sxy_gaussian, sxy_gaussian),
        shape=(h, w)
    )
    d.addPairwiseEnergy(pairwise_energy_gaussian, compat=compat_gaussian)

    # Perform inference
    Q = d.inference(n_iters)

    # Reorganize output
    srm_prob = np.array(Q).reshape((n_classes, h, w))
    return srm_prob


def apply_srm_to_batch(batch, avg_prob):
    srm_refined_prob_list = []
    for i in range(avg_prob.size(0)):
        # Prepare image data (H, W, 3) [0-255]
        img_np = (batch['query_img'][i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Prepare probability data (2, H, W)
        prob_np = avg_prob[i].numpy()

        # Apply SRM refinement
        srm_prob = dense_srm_refinement(
            img_np,
            prob_np,
            sxy_bilateral=65,      # Lower values retain more detail
            srgb_bilateral=10,     # Lower values make color more sensitive
            sxy_gaussian=3,
            compat_bilateral=7,    # Lower values reduce boundary overfitting
            compat_gaussian=5,     # Higher values increase regional consistency
            n_iters=5
        )
        srm_refined_prob_list.append(torch.from_numpy(srm_prob).unsqueeze(0))

    final_prob = torch.cat(srm_refined_prob_list, dim=0)
    return final_prob
