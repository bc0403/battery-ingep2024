# battery-ingep2024
code for INGEP2024 conference paper:

Zhu, G., Lv, X., Zhang, Y., Lu, Y., Han, G., Chen, M., Zhou, Y., & Jin, H. (2024). Estimating lithium-ion battery capacity from relaxation voltage using a machine learning approach. Journal of Physics: Conference Series, 2809(1), 012053. https://doi.org/10.1088/1742-6596/2809/1/012053

## Environment
* install Python 3.11.5
* create a virtual environment `python3.11 -m venv venv` and activate it `source ./venv/bin/activate`
* install Pytorch dev (for using mps [1]) `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`
* Install other packages `pip install -r requirements.txt`

## Files Description
* `00-get_database.sh`, download the dataset if it is not exist;
* `01-EDA-fig1a.py`, Exploratory Data Analysis (EDA) of dataset and plot fig1a
* `02-features_extraction.py`, extract features from dataset, relaxation voltage as one feature
* `03-data_prepare.py`, prepare data for algorithm, relaxation voltage as time series features
* `04-EDA-fig1b.py`, for fig1b
* `05-1dcnn.py`, 1d cnn for regression
* `06-results-fig345.py`, for fig3, fig4, and fig5

## References
[1] https://developer.apple.com/metal/pytorch/


