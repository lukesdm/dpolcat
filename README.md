# dpolcat

*dpolcat* is a tool for the semantic categorization of dual-polarimetric synthetic aperture radar (SAR) imagery. (Specifically, Sentinel-1 with VV and VH polarizations at present.)

It is inspired by Andrea Baraldi's [Satellite Image Automatic Mapper™](http://siam.andreabaraldi.com/) (SIAM™) concept of *spectral categories* for optical data, and the [Sen2Cube](https://www.sen2cube.at/) project.

Status: Effectively at prototype/proof of concept stage, under active development.


## Environment

Development and processing are supported within a [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) Jupyter Python environment.

These environments are based on [Pangeo](https://github.com/pangeo-data/pangeo-docker-images). This may be an option for use on other platforms, though connection to the EO imagery archive would need to be changed accordingly.


## Contents
#### 📄 `dpolcat.py`

The *dpolcat* module, containing the main algorithms/functions for performing polarimetric categorization.

#### 📄 `dpolcat_demo.ipynb`

A Jupyter Notebook demonstrating the use of *dpolcat*, including a simple end-to-end flood mapping example.

#### 📄 `dpolcat_perf.ipynb`

A Jupyter Notebook for measuring the computational and memory performance of *dpolcat* processing.

#### 📄 `dpolcat_proto.ipynb`

A Jupyter Notebook with the initial design and prototyping of the categorizer algorithms. It features a number of experiments.

#### 📁 `example_duisburg`

Supplementary folder for the demo notebook's flood mapping example, containing a QGIS project and associated data for accuracy assessment.


#### 📄 `poster.pdf`

A simple poster about the project presented internally.


## Credits

Imagery: Contains modified Copernicus Sentinel data, processed by ESA and others.

Flood reference: [Copernicus Emergency Mapping EMSR517](https://emergency.copernicus.eu/mapping/ems-product-component/EMSR517_AOI06_DEL_MONIT01_r1_RTP03/1)

## Authors

Created by Luke McQuade at [Z_GIS](https://www.plus.ac.at/geoinformatik/?lang=en), as part of the the Applied Geoinformatics MSc programme.

## Acknowledgements

We would like to greatly thank the members of the [EO Analytics Group](https://www.plus.ac.at/geoinformatik/research/research-areas/eo-analytics/?lang=en) and Dr. Zhara Dabiri of the [Risk, Hazard and Climate Lab](https://www.plus.ac.at/geoinformatik/research/research-areas/risk-hazard-climate/?lang=en) for their ideas and support, and Assoc. Prof. Hermann Klug and Dr. Bernhard Zagel for their instruction and continuous feedback.