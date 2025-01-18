# run this script to perform unittests on the modules
# very basic module
python3 utils.py -v --only-automatic-test-workflow=on

# basic modules
python3 structure.py -v --only-automatic-test-workflow=on
python3 abacus.py -v --only-automatic-test-workflow=on
python3 deltaspin.py -v --only-automatic-test-workflow=on

# advanced modules
python3 spin_flip_const_vol.py -v --only-automatic-test-workflow=on
python3 spin_flip_scan_vol.py -v --only-automatic-test-workflow=on
python3 phonon.py -v --only-automatic-test-workflow=on
python3 magmom_exch_const.py -v --only-automatic-test-workflow=on
python3 magnon_spectrum.py -v --only-automatic-test-workflow=on

# the elasticity module needs pymatgen, if there is no pymatgen installed, the test will be skipped
conda list > unittest_conda_installed_packages.txt
if grep -q pymatgen unittest_conda_installed_packages.txt; then
    python3 elastic.py -v --only-automatic-test-workflow=on
else
    echo "WARNING: pymatgen is not installed, skipping the test for elastic.py"
fi
rm unittest_conda_installed_packages.txt
