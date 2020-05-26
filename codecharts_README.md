# CodeCharts1K Dataset

Thank you for downloading the CodeCharts1K dataset! 

This zip contains the following directory structure: 
```
.
├── fix_coords
│   ├── 3000 [1000 entries exceeds filelimit, not opening dir]
│   ├── 500 [1000 entries exceeds filelimit, not opening dir]
│   └── 5000 [1000 entries exceeds filelimit, not opening dir]
├── fix_coords_accum
│   ├── 3000 [1000 entries exceeds filelimit, not opening dir]
│   ├── 500 [1000 entries exceeds filelimit, not opening dir]
│   └── 5000 [1000 entries exceeds filelimit, not opening dir]
├── fix_maps
│   ├── 3000 [1000 entries exceeds filelimit, not opening dir]
│   ├── 500 [1000 entries exceeds filelimit, not opening dir]
│   └── 5000 [1000 entries exceeds filelimit, not opening dir]
├── fix_maps_accum
│   ├── 3000 [1000 entries exceeds filelimit, not opening dir]
│   ├── 500 [1000 entries exceeds filelimit, not opening dir]
│   └── 5000 [1000 entries exceeds filelimit, not opening dir]
├── heatmaps
│   ├── 3000 [1000 entries exceeds filelimit, not opening dir]
│   ├── 500 [1000 entries exceeds filelimit, not opening dir]
│   └── 5000 [1000 entries exceeds filelimit, not opening dir]
├── heatmaps_accum
│   ├── 3000 [1000 entries exceeds filelimit, not opening dir]
│   ├── 500 [1000 entries exceeds filelimit, not opening dir]
│   └── 5000 [1000 entries exceeds filelimit, not opening dir]
├── raw_img [1000 entries exceeds filelimit, not opening dir]
└── splits.json
```

with the following contents: 
- **`splits.json`: 4-split train-test-val breakdown** 
- **`raw_img`: raw image files**
- **`heatmaps`: multi-duration saliency heatmaps**
- `fix_maps`: binary fixation maps where gaze point locations are set to 1, else 0 
- `fix_coords`: image coordinates of gaze point locations 
- `heatmaps_accum`: heatmaps where gaze points are accumulated over durations; not used in paper
- `fix_maps_accum`: fixation maps where gaze points are accumulated over durations; not used in paper
- `fix_coords_accum`: image coordinates where gaze points are accumulated over durations; not used in paper
