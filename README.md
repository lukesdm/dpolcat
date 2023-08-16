# dpolcat

*dpolcat* is a tool for the semantic categorization of dual-polarimetric synthetic aperture radar (SAR) imagery. (Specifically, Sentinel-1 with VV and VH polarizations at present.)

Status: Effectively at prototype/proof of concept stage, under active development.


## Environment

Development and processing are supported within a [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) Jupyter Python environment.

These environments are based on [Pangeo](https://github.com/pangeo-data/pangeo-docker-images). This may be an option for use on other platforms, though connection to the EO imagery archive would need to be changed accordingly.


## Contents
### 📄 `dpolcat.py`

The *dpolcat* module, containing the main algorithms/functions for performing polarimetric categorization.

#### 📄 `dpolcat_demo.ipynb`

A Jupyter Notebook demonstrating the use of *dpolcat*, including a simple end-to-end flood mapping example.

#### 📄 `dpolcat_perf.ipynb`

A Jupyter Notebook for measuring the computational and memory performance of *dpolcat* processing.

#### 📄 `dpolcat_proto.ipynb`

A Jupyter Notebook with the initial design and prototyping of the categorizer algorithms. It features a number of experiments.

#### 📁 `example_duisburg`

Supplementary folder for the demo notebook's flood mapping example, containing a QGIS project and associated data for accuracy assessment.


## Credits

Imagery: Contains modified Copernicus Sentinel data, processed by ESA and others.

Flood reference: [Copernicus Emergency Mapping EMSR517](https://emergency.copernicus.eu/mapping/ems-product-component/EMSR517_AOI06_DEL_MONIT01_r1_RTP03/1)
