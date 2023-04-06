#!/usr/bin/env python
"""
We dont use console for this one, but run it as a normal python script. Based on a learned sdf, it extracts a discrete volume in which each point has a sd value.
The result is used for evaluation the volume computation accuracy, i.e. the IOU.

Specifically, based on the camera locations, as well as how sdfstudio data is generated, we 
    1) discretize the volume according to a grid of special resolution
    2) for each point, first normalize its location to the normalized space,
    2) evaluate each the sd on the grid, and thresh them to occupancy
    3) save the results.
    
We can also address the point cloud in the same way, for which we dont need to normalize the location.
Specifically, provided a point cloud with 3D locations, we 
    1) check whether a point is in the grid. If yes, then =1, else 0
    2) save the results.
    
Afterwards, it would be straightforward to compare two volumes according to the IOU.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import tyro
from rich.console import Console
import json
import os
import pickle


from nerfstudio.model_components.ray_samplers import save_points
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import (
    get_surface_occupancy,
    get_surface_sliding,
    get_surface_sliding_with_contraction,
)

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")



@dataclass
class ExtractVolume:
    
    # Path to config YAML file.
    load_config: Path
    # space resolution. i.e. we discretize the space into [res, res, res]
    resolution: int = 128
    # Name of the output file.
    # output_path: Path = Path("output.pkl")

    
    def main(self) -> None:
        """Main function."""
        self.output_path = '/'+os.path.join(*(str(self.load_config).split('/')[:-2] + ['vol.pkl']))
        assert str(self.output_path)[-4:] == ".pkl"

        ccgg, pipeline, _ = eval_setup(self.load_config)
        datapath = os.path.join(str(ccgg.pipeline.datamanager.dataparser.data), 'meta_data.json')
        with open(datapath) as f:
            dataconfig = json.load(f)
        transf_gt2world = torch.linalg.inv(torch.tensor(dataconfig['worldtogt'])) # the transformation function from the original space to the normalized space
        xmin= ymin= zmin = -1
        xmax= ymax= zmax = 1

        """create the volume"""
        xx = torch.linspace(xmin, xmax, self.resolution).cuda()
        yy = torch.linspace(ymin, ymax, self.resolution).cuda()
        zz = torch.linspace(zmin, zmax, self.resolution).cuda()
        grid_x, grid_y, grid_z = torch.meshgrid(xx,yy,zz, indexing='ij')
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)],
                             dim=-1)
        
        sdfs = pipeline.model.field.forward_geonetwork(points)[:, 0].contiguous().view(self.resolution, self.resolution, self.resolution)
        occ = -sdfs
        occ[occ>=0] = 1
        occ[occ<0] = 0
        
        results = {}
        results['occupancy'] = occ.detach().cpu().numpy()
        results['normalize_mat'] = transf_gt2world.detach().cpu().numpy()
        with open(self.output_path, 'wb') as f:
            pickle.dump(results, f)
        

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractVolume]).main()


if __name__ == "__main__":
    entrypoint()