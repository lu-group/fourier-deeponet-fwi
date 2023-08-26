# Fourier-DeepONet for full waveform inversion (FWI)

The data and code for the paper [M. Zhu, S. Feng, Y. Lin, & L. Lu. Fourier-DeepONet: Fourier-enhanced deep operator networks for full waveform inversion with improved accuracy, generalizability, and robustness. Computer Methods in Applied Mechanics and Engineering, 416, 116300, 2023](https://doi.org/10.1016/j.cma.2023.116300)

## Datasets

- [FlatVel-B (FVB)](data/fvb)
- [CurveVel-A (CVA)](data/cva)
- [CurveFault-A](data/cfa)
- [Style-A](data/sta)

Run data_gen_f.py, data_gen_loc.py, and data_gen_loc_f.py to generate seismic data of FWI-F, FWI-L, and FWI-FL, respectively.

## Code
In [train.py](src/train.py) and [test.py](src/test.py), change the arguements 'dataset' and 'task' in main function as needed.

Run [train.py](src/train.py) for training, and then run [test.py](src/test.py) for testing.

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{zhu2023fourier,
  title   = {Fourier-DeepONet: Fourier-enhanced deep operator networks for full waveform inversion with improved accuracy, generalizability, and robustness},
  author  = {Zhu, Min and Feng, Shihang and Lin, Youzuo and Lu, Lu},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {416},
  pages   = {116300},
  year    = {2023}
  doi     = {https://doi.org/10.1016/j.cma.2023.116300}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
