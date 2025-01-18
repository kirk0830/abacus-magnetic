# DeltaSpin machine-learning model verification toolkits
## Files
The correspondence between file name and the code of figure is listed below
|file name|figure|
|---|---|
|spin_flip_scan_vol.py|6(b)|
|spin_flip_const_vol.py|6(c)|
|magmom_exch_const.py|7(a)|
|magnon_spectrum.py|7(b)|
|phonon.py|S?|

*NOTE*: There is also a code to calculate the elastic properties, by `elastic.py`.

There are files in which basic functions are implemented, for user do not need to take care, they are: 
- utils.py
- abacus.py
- deltaspin.py

## Usage
Please read the leading instruction in each file. All codes can be run and produce figure directly after a proper assignment of parameters.

For each code, please always read the lines in 
```python
if __name__ == '__main__':
    # ...
```
block.

*NOTE*: before running all these codes, please make sure the proper conda virtual environment has been activated. Not:
```
(base) /path/to/your/work/folder:~
```
Instead, 
```
(DeltaSpin_devel) /path/to/your/work/folder:~
```
. To activate the environment with specific name (e.g., `DeltaSpin_devel`), in your prompt:
```
conda activate DeltaSpin_devel
```
