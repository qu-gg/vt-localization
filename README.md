<h1 align='center'>A Hybrid Machine Learning Approach to Localizing the Origin of Ventricular Tachycardia Using 12-Lead Electrocardiograms</h1>
<h2 align='center'>[<a>arXiv</a>, <a href='https://cslide-us.ctimeetingtech.com/hrs20/attendee/eposter/poster/956https://cslide-us.ctimeetingtech.com/hrs20/attendee/eposter/poster/956'>HRS Presentation</a>]</h2>

<p>This repository holds the experiments and models as explored in the work, "A Hybrid Machine Learning Approach to Localizing the Origin of Ventricular Tachycardia Using 12-Lead Electrocardiograms." It separates the runtimes of the population and patient-specific models with notes on how to run each.</p>

<h2>Running</h2>
<p>Due to confidentiality agreements with our collaborators, the data used in training and testing the models cannot be made public. However, the data processing scripts are available to format ECG datasets into runnable forms. Modifications will need to be made in the processing scripts when applying custom data, due to initial data formatting.</p>

<h3>Population Models</h3>
<p>Contains the tested population models, including Linear Regression, CNN, and a CNN-VAE. Details on the setup of each are given within the paper.</p>

<p>HOW TO RUN</p>

<h3>Patient-Specific Models</h3>
<p>Contains the patient-specific models that were tested, as well as the experiments to compare them. The test runtime expects 4 initial files in the given folder:</p>
  
  ```
  data/
  ├── coords.csv   - Cartesian coordinates of the pacing sites. Shape [N, 3].
  ├── data.csv     - Raw ECGs of each pacing site, with the leads concatenated horizontally. Shape [N, 12 * Length of a lead].
  ├── points.csv   - Represents the population model's segment predictions alongside the true segment. Shape [N, 6]
  ├── segments.csv - Holds the 16-segment label for each patient
  
  Generated via pat_spec_pipeline.py
  ├── sampled/
      └── samp_files - sampled beats from each pacing site
  ├── patient-datasets/
      └── folders/ - holds the separated data per patient to iterate over in files
  ```
  
<p>The experiment file is 'combined-test.py,' which handles running through all the models on the same initialized sets of points for each pacing site. Configuring hyperparameters is available in 'confi.py.'</p>
  
<h3>Results</h3>
<p>Results of each experiment are output to the terminal, with formatted results of each stat tracked.</p>

<h3>Contact</h3>
<p>Feel free to reach out with any questions or comments. Contact details are available within the publication.</p>
