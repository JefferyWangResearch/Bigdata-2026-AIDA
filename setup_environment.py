"""Print or install the historical analysis dependencies in an active virtual environment."""
import argparse
import subprocess
import sys

PACKAGES = ['certifi==2026.7.22', 'cftime==1.6.4.post1', 'charset-normalizer==3.5.2', 'click==8.1.8', 'contourpy==1.3.0', 'cycler==0.12.1', 'fonttools==4.60.2', 'idna==3.20', 'importlib_resources==6.5.2', 'joblib==1.5.3', 'kiwisolver==1.4.7', 'mapbox-vector-tile==2.2.0', 'matplotlib==3.9.4', 'mercantile==1.2.1', 'netCDF4==1.7.2', 'numpy==2.0.2', 'packaging==26.3', 'pandas==2.3.3', 'patsy==1.0.3', 'pillow==11.3.0', 'protobuf==6.33.6', 'pyclipper==1.3.0.post6', 'pyparsing==3.3.3', 'pyproj==3.6.1', 'pyshp==3.1.6', 'python-dateutil==2.9.0.post0', 'pytz==2026.5', 'requests==2.32.5', 'scikit-learn==1.6.1', 'scipy==1.13.1', 'shapely==2.0.7', 'six==1.17.0', 'statsmodels==0.14.6', 'tabulate==0.9.0', 'threadpoolctl==3.7.0', 'tzdata==2026.5', 'urllib3==2.6.3', 'zipp==3.23.1']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    if not args.install:
        print('\n'.join(PACKAGES))
        return
    if sys.prefix == sys.base_prefix:
        parser.error('Activate a virtual environment before installing dependencies.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', *PACKAGES], check=True)

if __name__ == '__main__':
    main()
