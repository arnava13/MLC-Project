# February 25 2025
Searched some related datasets.
### Datasets to Use
#### Corporate Climate Reporting Data
CDP Climate Change Data: Data submitted by companies regarding greenhouse gas (GHG) emissions, energy use, and risk assessments. https://www.cdp.net/en (request access)
#### Corporate Sustainability Reports: Specific GHG emissions and renewable energy usage ratios from companies like Unilever, Coca-Cola, and Microsoft.
Unilever: https://www.unilever.com/sustainability/responsible-business/sustainability-performance-data/　<br>
Coca-cola: https://www.coca-colacompany.com/content/dam/company/us/en/reports/2023-environmental-update/2023-environmental-update.pdf <br>
Microsoft: https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/msc/documents/presentations/CSR/Microsoft-2024-Environmental-Sustainability-Report.pdf
#### Remote Sensing Data
OCO-2: CO₂ concentration data taken from satellite images.
https://ocov2.jpl.nasa.gov/science/oco-2-data-center/ <br>
GHGSat: Satellite data monitoring greenhouse gas emissions (methane and CO₂) from specific locations.
https://earth.esa.int/eogateway/missions/ghgsat#data-section (required approval) 

# February 26 2025
### Major difficulty
Some useful datsets require approval from the data provider. <br>
How to adjust data from reports with remote sensing data? <br>
It is difficult to definitively determine whether it is due to the company's location or the nature of its business.<br>
It seems very hard to distinguish the impact of a company's business on the environment due to the geoggraphic scales and other environmental factors.

### Possible directions
Anomaly detection of CO₂ concentration data from reports by using remote sensing data. <br>
Time series forecasting of CO₂ concentration<br>
Evaluate ESC score of companies by using reports and remote sensing data.

### first meeting
Discuss the direction of the project. <br>
Change our plan to follow some public projects. <br>
EY - The 2025 EY Open Science AI and Data Challenge: Cooling Urban Heat Islands: https://challenge.ey.com/challenges/the-2025-ey-open-science-ai-and-data-challenge-cooling-urban-heat-islands-external-participants/data-description <br>

#  February 28 2025
### Datasets
European Sentinel-2 optical satellite data <br>
NASA Landsat optical satellite data <br>

### Additional Datasets
Building footprints of the Bronx and Manhattan regions <br>
Detailed local weather dataset of the Bronx and Manhattan regions on 24 July 2021

# March 8 2025
## Week 1: Planning

Output Format:

1123 Grid Cells

60 time stamps

Time series model – 1123x1 

Over 60 timestamps

## Input Data:

**Satellite Data**

Satellite images of the area in question, used to derive NDVI, NDWI, NDBI and LST.

Generate median satellite images for our 60 timestamps, ensuring a imaging region that exactly matches the coordinates of our output dataset.

Use NDVI, NDWI, NDBI, LST as our 4 channels and run a convolution over the $W\times H \times4$ image.  (start with this)

“Participants might explore other combinations of bands from the Sentinel-2 and from other satellite datasets as well. For example, you can use mathematical combinations of bands to generate various indices </a> which can then be used as features in your model. These bands or indices may provide insights into surface characteristics, vegetation, or built-up areas that could influence UHI patterns.” 

– Perhaps we use the spectral bands directly instead of NDVI, NDWI, NDBI, LST and these can be our channels?? There are a lot so we would run ablation, starting with those used for NDVI, NDWI, NDBI and LST and then expanding outwards

“Instead of a single point data extraction, participants might explore the approach of creating a focal buffer around the locations (e.g., 50 m, 100 m, 150 m etc). For example, if the specified distance was 50 m and the specified band was “Band 2”, then the value of the output pixels from this analysis would reflect the average values in band 2 within 50 meters of the specific location. This approach might help reduction in error associated with spatial autocorrelation. In this demonstration notebook, we are extracting the band data for each of the locations without creating a buffer zone.” – We should use the resolution of the output dataset i.e **gridcell size** as the area of averaging

**Weather** 

2x locations Bronx & Manhattan

- When predicting for a grid cell, use weather from the closest station

5x columns, merge to 60 timestamps if there are more

—> 5D tensor for every timestamp

**Building Footprint**

- Polygons for buildings
- Perhaps add an embedding of these to the embedding of the final grid cells, so that we can match them exactly on gridcells

**Traffic**

- Research shows car traffic results in UHI, so we are going to use https://developers.google.com/maps/documentation/javascript/examples/layer-traffic to get a scalar traffic density value for each grid cell. Add/concat at near the end of the model when we have grid cells.