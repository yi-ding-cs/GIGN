# GIGN
This is the PyTorch implementation of GIGN for continuse emotion recognition using MAHNOB-HCI dataset

[Yi Ding, Cuntai Guan, “GIGN: Learning Graph-in-graph Representations of EEG Signals for Continuous Emotion Recognition”, in EMBC, 2023 (Oral)](https://ieeexplore.ieee.org/document/10340644).

# Prepare the data
Download MAHNOB-HCI dataset [here](https://mahnob-db.eu/hci-tagging/). And set the data folder as the root_directory in configs.py, e.g., /home/dingyi/MAHNOB/. This folder should contains two subfolders, ./Sessions/ and ./Subjects/.

Get the continuous label in this [repo](https://github.com/soheilrayatdoost/ContinuousEmotionDetection). Put the lable_continous_Mahnob.mat at the data folder, e.g., /home/dingyi/MAHNOB/lable_continous_Mahnob.mat

Note that it might pop some error messages when you create the dataset by using generate_dataset.py. It is because there are some format errors in the original data. You can identify the file according to the error message and correct the format error in that file. 

# Run the code
Setp 1: You need to check the config.py first, changing the parameters accordingly.

Step 2: Run generate_dataset.py.

Step 3: Check the parameters in the main.py, changing them accordingly.

Step 4: Run main.py to train and evaluate the network.

Step 5: Using generate_results_csv.py to get the summarized results.

## Example of using GIGN
```python
from model.GiG import GiG

data = torch.randn(1, 96, 32, 6)  # (batch_size=1, data_sequence=96, EEG_channel=32, feature=6)

# For regression, the output is (batch_size, data_sequence, 1).
net = GiG(
          layers_graph=[2, 2],
          num_chan=32,
          num_seq=96,
          num_feature=6,
          hidden_graph=128,
          K=[3, 4],
          dropout=0.5,
          num_class=1,
          encoder_type='Cheby'
          )
preds = net(data)
```
# Cite
Please cite our paper if you use our code in your own work:

```
@INPROCEEDINGS{10340644,
  author={Ding, Yi and Guan, Cuntai},
  booktitle={2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 
  title={GIGN: Learning Graph-in-graph Representations of EEG Signals for Continuous Emotion Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Emotion recognition;Codes;Network topology;Biological system modeling;Brain modeling;Electroencephalography;Graph neural networks},
  doi={10.1109/EMBC40787.2023.10340644}}

```
# CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |
