CC Model (Correlation Coefficient):
Correlation Coefficient model based on some active guidance principles.
The active guidance aims to intelligently suggest sites such as to explore different regions of the heart if current
predictions are inaccurate, and to exploit the current region as we move closer to the target.
To explore the unknown pacing space, we focus on the area about which the current model is the least certain.
This is determined by the training site with the highest localization error when predicted by a model built using the
rest of the training samples. In addition, the current prediction is also suggested as a second pacing site.
In the case that the two suggested sites are within 5-mm (approximate diameter of typical ablation lesion) with each
other, only the model-predicted site will be suggested.

The CC portion of the model relates to trying to select the training site of highest error within a certain CC
threshold to the target site. So for the pacing site to be suggested as an area of interest, it must be one that
has the highest localization error given by the rest of the training samples and be above a certain CC threshold.


RS Model (Random Selection):
Simulates as if the clinician were to just randomly pace on the heart until it finds a pacing site
within an acceptable tolerance to the target site


UtilFuncs.py:
Simply holds some common base functions that all of the models use
