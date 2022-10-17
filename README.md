# Classification of Forest Cover type using Neural Networks

### Dataset Description:

The dataset for classification considered was downloaded from the
University of California Irvine, Machine Learning Repository titled
‘The Forest Covertype Data’. This dataset was used in predicting
forest cover types in undisturbed forests. The data set can be access from [here](https://archive.ics.uci.edu/ml/datasets/covertype)

This dataset consists of cartographic variables only and no remotely
sensed data. The actual forest cover type for a given observation (30 x
30-meter cell) was determined from US Forest Service (USFS) Resource
Information System (RIS) data. Independent variables were derived from
data originally obtained from US Geological Survey (USGS) and USFS data.
Data is in raw form and contains binary columns of data for qualitative
independent variables such as wilderness areas and soil types.

The study area consists of four wilderness areas found in the
Roosevelt National Forest of northern Colorado. These four wilderness
areas include: Neota (area 2) probably has the highest mean elevational
value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area
3) would have a lower mean elevational value, while Cache la Poudre
(area 4) would have the lowest mean elevational value.

Seven primary major tree species in these areas have been identified.
Their names and their class distributions are shown below.

| Types of Trees    | Class | Number of Records |
| --------------------- | --------- | --------------------- |
| Spruce – Fir      | 1     | 211,840           |
| Lodgepole Pine    | 2     | 283,301           |
| Ponderosa Pine    | 3     | 35,754            |
| Cottonwood/Willow | 4     | 2,747             |
| Aspen             | 5     | 9,493             |
| Douglas Fir       | 6     | 17,367            |
| Krummholz         | 7     | 20,510            |
|                   | Total | 581,012   |

Number of Attributes include 12 measures, but 54 columns of data (10
quantitative variables, 4 binary wilderness areas, and 40 binary soil
type variables). Information about these attributes is given in the
table below. The order of the attributes in the dataset is also shown in
the table.

| Attribute (Order in the dataset)            | Data Type    | Measurement                 | Description                                             |
| ----------------------------------------------- | ---------------- | ------------------------------- | ----------------------------------------------------------- |
| Elevation (0.)                              | quantitative | meters                      | Elevation in meters                                     |
| Aspect (1.)                                 | quantitative | azimuth                     | Aspect in degrees azimuth                               |
| Slope (2.)                                  | quantitative | degrees                     | Slope in degrees                                        |
| Horizontal\_Distance\_to\_Hydrology (3.)    | quantitative | meters                      | Horizontal distance to nearest surface water features   |
| Vertical\_Distance\_To\_Hydrology (4.)      | quantitative | meters                      | Vertical distance to nearest surface water features     |
| Horizontal\_Distance\_to\_Roadways (5.)     | quantitative | meters                      | Horizontal distance to nearest roadway                  |
| Hillshade\_9am (6.)                         | quantitative | 0 to 255 index              | Hillshade index at 9am, summer solstice                 |
| Hillshade\_Noon (7.)                        | quantitative | 0 to 255 index              | Hillshade index at noon, summer solstice                |
| Hillshade\_3pm (8)                          | quantitative | 0 to 255 index              | Hillshade index at 3pm, summer solstice                 |
| Horizontal\_Distance\_To\_Fire\_Points (9.) | quantitative | meters                      | Horizontal Distance to nearest wildfire ignition points |
| Wilderness\_Area (4 binary columns) (10-13) | qualitative  | 0 (absence) or 1 (presence) | Wilderness area designation                             |
| Soil\_Type (40 binary columns) (14-53)      | qualitative  | 0 (absence) or 1 (presence) | Soil Type Designation                                   |
| Cover\_Type (7 types) (54)                  | qualitative  | 1 to 7                      | Forest Cover Type Designation                           |

Soil Types: 40 binary columns. This is based on the USFS Ecological
Land type Units (ELUs) for this study area. Descriptions of the 40 soil
types can be found in Blackard and Dean (2000).

Basic Summary Statistics for quantitative variables are shown below:

| Attribute (Units)                                       | Mean    | Standard Deviation |
| ----------------------------------------------------------- | ----------- | ---------------------- |
| Elevation (meters)                                      | 2959.36 | 279.98             |
| Aspect (azimuth)                                        | 155.65  | 111.91             |
| Slope (degrees)                                         | 14.10   | 7.49               |
| Horizontal\_Distance\_To\_Hydrology (meters)            | 269.43  | 212.55             |
| Vertical\_Distance\_To\_Hydrology (meters)              | 46.42   | 58.30              |
| Horizontal\_Distance\_To\_Roadways (meters)             | 2350.15 | 1559.25            |
| Hillshade\_9am (0 to 255 index)                         | 212.15  | 26.77              |
| Hillshade\_Noon (0 to 255 index)                        | 223.32  | 19.77              |
| Hillshade\_3pm (0 to 255 index)                         | 142.53  | 38.27              |
| Horizontal\_Distance\_To\_Fire\_Points (0 to 255 index) | 1980.29 | 1324.19            |

### Scope:

The scope of this assignment is to apply Multi-Layered Perceptron (MLP),
a deep learning model, to this classification problem and eventually
predict forest cover types. Specifically, three models
MLP<sub>low</sub>, MLP<sub>high</sub>, MLP<sub>long</sub> (includes
auto-encoder) are to be employed on this dataset, and differences among
these models are evaluated in terms of their performances and accuracies
obtained on Train sets and Test Sets.

### Data Preprocessing:

There are a total of 581,012 observations in this dataset. Due to a
large variance of instances among different classes, under-sampling and
oversampling techniques were implemented to bring uniformity among the
classes. Classes 1, 2, and 3 were under-sampled using a near-miss
function whereas classes 4, 5, and 6 were oversampled using SMOTE
function.

| Class | Number of Records | Number of Records after Data Processing |
| --------- | --------------------- | ------------------------------------------- |
| 1     | 211,840           | 20,510                                  |
| 2     | 283,301           | 20,510                                  |
| 3     | 35,754            | 20,510                                  |
| 4     | 2,747             | 20,510                                  |
| 5     | 9,493             | 20,510                                  |
| 6     | 17,367            | 20,510                                  |
| 7     | 20,510            | 20,510                                  |
| Total | 581,012           | 143,570                                 |

### Model Building and Model Evaluation:

The data set is first randomly split into Train and Test sets of sizes
80% and 20% of the whole dataset respectively. After splitting, the
train set was fed to the input layer and subsequently into a hidden
layer with 54 perceptrons (number of features = 54), followed by an
output layer with 7 perceptrons. ADAMS optimizer was used here for all
MLP models for backpropagation and Google Colab was used to write and
run the code.

### Simple MLP:

Assumptions:

1.  Batch size = Size of Train Set/50

2.  Epochs = 10

The activation function applied for the input layer and hidden layer was
ReLU. SOFTMAX activation function was applied on the output layer and
probabilities of the predicted class were computed for each instance.
This predicted class for each instance would have the highest
probability among the 7 classes. One-hot encoding was performed for each
case to our class attribute in the dataset (Attribute No. 54) and was
treated as true/actual output. Finally, the Categorical Cross Entropy
function was applied to compute losses at each Epoch for Train and Test
sets. These losses attained at each epoch are Average Cross Entropies
(AVCREs). See the plot below for AVCRE vs. Epochs for a Train set and
Test set.

![Chart Description automatically generated](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture1.png)

From the figure above, we observe that the losses for train and test
sets fall steeply from 59 and 12 respectively, after two Epochs and
reach a plateau. The attained losses, which are the AVCREs, were
transformed into probabilities (P = e <sup>- AVCRE</sup>) and plotted
against Epochs below.

![Chart, line chart Description automatically
generated](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture2.png)

At the end of the last Epoch, the weights and thresholds were fixed and
MLP was applied, and losses were computed. AVCREs on each class is
computed as probabilities vs Epochs

| Class | AVCRE     |              |
| --------- | ------------- | ------------ |
|           | Train Set | Test Set |
| 1         | 0.360         | 0.361        |
| 2         | 0.195         | 0.193        |
| 3         | 0.329         | 0.335        |
| 4         | 0.097         | 0.097        |
| 5         | 0.113         | 0.112        |
| 6         | 0.224         | 0.230        |
| 7         | 0.169         | 0.171        |

From the table above, we observed that the losses obtained were very low
indicating that the model was predicting the classes (forest cover
types) well.

### MLP<sub>low</sub>:

The entire dataset has four attributes with zero values for all
instances, which were dropped as part of data preprocessing before the
standardization and application of Principal Component Analysis (PCA).
PCA was performed on the new dataset to reduce the number of attributes
based on 90% Principal of Explained Variance (PEV). PEVs vs index of the
eigenvalues obtained from the correlation matrix were plotted below.

![Chart, line chart Description automatically
generated](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture3.png)

From the plot above, the number of attributes obtained for 90% PEV is
36.

h<sub>low</sub> = 36

The hidden layer in our MLP was now changed to contain only 36
perceptrons and the automatic learning using this new MLP<sub>low</sub>
was launched.

Assumptions:

1.  Batch Size = 100

2.  Epochs = 50

The AVCREs obtained using the MLP<sub>low</sub> model are plotted
against Epochs as shown below.

![Chart, histogram Description automatically
generated](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture4.png)

StabTrain: The epoch of stabilization for the above curve would be
22.

MinTest: The epoch for which MinTest to be satisfied would be 44.

SafeZone: The possible values of m are
\[3,4,5,7,8,9,12,14,15,23,25,26,27,28,30,32,45,46,47,48,49,50\].

SafeMinTest: The m\* would be 25 as the testAVCRE reaches its
minimum on the safe zone and then increases steadily.

The value of mSTOP was fixed as 25 and then the model was relaunched
with the number of epochs set to mSTOP.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture5.png)

Confusion matrix for the train set

From the confusion matrix computed for the train set above, we observed
that all classes had high prediction accuracies ranging from 84 to 95%,
except for class 1 at 66%. For class 1, 21% of the observations were
misclassified as class 2, and 14% were misclassified as class 7. This
model was able to predict class 5 with the highest accuracy at 95%,
while only 0.12% of the observations were misclassified as class 2, 3.4%
as class 3, and 1.9% as class 6.

| Class | Percentage of Correct Classification |
| --------- | ---------------------------------------- |
| 1         | 66%                                      |
| 2         | 87%                                      |
| 3         | 88%                                      |
| 4         | 94%                                      |
| 5         | 95%                                      |
| 6         | 84%                                      |
| 7         | 85%                                      |

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture6.png)

Confusion matrix for test set

The confusion matrix generated on the test set also gave similar results
as that of the train set. The model was able to predict all classes well
with 83 to 95% accuracy except class 1. Class 1 was most misclassified
at 65% accuracy while class 5 had an accuracy of 95%. For class 1, 21%
of the observations were misclassified as class 2, and 14% were
misclassified as 7. In the case of class ‘5’, only 0.17% of the
observations were misclassified as class 2, 3.1% as class 3, and 2.1% of
the observations as class 6.

| Class | Percentage of Correct Classification |
| --------- | ---------------------------------------- |
| 1         | 65                                       |
| 2         | 86                                       |
| 3         | 87                                       |
| 4         | 94                                       |
| 5         | 95                                       |
| 6         | 83                                       |
| 7         | 85                                       |

Computation time for this model: 197.95 sec

### MLP<sub>high</sub>:

For this model, initially, PCA analysis *was performed separately for
each true class of their respective instances. Note that before PCA
analysis was performed on all seven classes of instances, the attributes
with all zero entries are dropped and then standardized. PEV vs
principal component curves for all seven classes were plotted below and
‘h’ values were computed from them.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture7.png)

Class 1: 90% PEV = 22, so h<sub>1</sub> = 22

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture8.png)

Class 2: 90% PEV = 25, so h<sub>2</sub> = 25

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture9.png)

Class 3: 90% PEV = 19, so h<sub>3</sub> = 19

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture10.png)

Class 4: 90% PEV = 15, so h<sub>4</sub> = 15

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture11.png)

Class 5: 90% PEV = 22, so h<sub>5</sub> = 22

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture12.png)

Class 6: 90% PEV = 22, so h<sub>6</sub> = 22

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture13.png)

Class 7: 90% PEV = 23, so h<sub>7</sub> = 23

h<sub>high</sub> = h<sub>1</sub> + h<sub>2</sub> + h<sub>3</sub> + h<sub>4</sub> + h<sub>5</sub> + h<sub>6</sub> + h<sub>7</sub> = 148

Using the h<sub>high</sub> value obtained above, the new model
MLP<sub>hgh</sub> with 148 neutrons in the hidden layer was
launched similar to previous model with the same batch size and Epochs.

Assumptions:

1.  Batch Size = 100

2.  Epochs = 50

The AVCREs were now computed and plotted against Epochs as shown below.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture14.png)

From the plot above, the losses start from lower values than Simple MLP
and MLP<sub>low</sub> for train and test sets at 8 and 4
respectively and drop down steeply until they reach a plateau at
approximately 20 Epochs.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture15.png)

Confusion matrix for the train set

From the confusion matrix above, this MLP<sub>high</sub> model
predicted classes for all instances in the train set with high
accuracies ranging from 85% to 95%. This model predicted better results
compared to the previous model.

| Class | Percentage of Correct Classification |
| --------- | ---------------------------------------- |
| 1         | 85                                       |
| 2         | 91                                       |
| 3         | 92                                       |
| 4         | 91                                       |
| 5         | 95                                       |
| 6         | 92                                       |
| 7         | 86                                       |

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture16.png)

Confusion matrix for test set

A similar trend as that of the train set was observed for the test set.
From the confusion matrix above, the MLP<sub>high</sub> was
able to predict classes for all instances in the test set with high
accuracy ranging from 84% to 94%.

| Class | Percentage of Correct Classification |
| --------- | ---------------------------------------- |
| 1         | 84                                       |
| 2         | 89                                       |
| 3         | 91                                       |
| 4         | 91                                       |
| 5         | 94                                       |
| 6         | 90                                       |
| 7         | 86                                       |

Computation Time for this model: 262.6 sec

Comparing the confusion matrices for train sets and test sets obtained
the best number of perceptrons to be used in our model would be
h<sub>high</sub> = 148

### MLP<sub>long</sub>:

Step 1: Auto-Encoder Construction

Initially, the entire dataset is transformed by the following equation
by using the weights and the biases from the MLP<sub>high</sub> model
hidden layer, and the resultant data is computed to construct an
auto-encoder algorithm.

_Z<sub>n</sub> = RELU(X<sub>n</sub> * W + B)_

Where X<sub>n</sub> = Vector of Case ‘n’ from the dataset,

W, B = Weights, and Biases computed from the hidden layer of the
MLP<sub>high</sub> model.

Z<sub>n</sub> = Transformed vector of Case ‘n’

The principal component analysis of the Z<sub>n</sub> results in the
size of the hidden layer to construct an autoencoder model and we found
out the initial 23 attributes of the Z<sub>n</sub> would give the 95%
PEV of the same dataset.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture17.png)

An auto-encoder model is constructed by 3 layers, the 1<sup>st</sup>
layer is used as an input layer with Z<sub>n</sub> dimensions, the
2<sup>nd</sup> layer has a dimension of 22 perceptrons which is
evaluated from the above PCA analysis, this compresses the Z<sub>n</sub>
data, which is encoded and eventually, this data is decoded by the
3<sup>rd</sup> layer with the same Z<sub>n</sub> dimensions. The
following is the sample structure of the auto-encoder model.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture18.png)

Assumptions:

1.  1<sup>st</sup> and 3<sup>rd</sup> layer: 141 Perceptron

2.  2<sup>nd</sup> layer: 22 Perceptron (PAC for Z<sub>n</sub>)

3.  Epochs: 50

4.  Batch Size: 100

In this model, the ReLU activation function was applied to all the
layers. The data was compressed in the hidden layer L<sub>2</sub>; the
hidden layer L<sub>3</sub> decoded the information from the hidden layer
L<sub>2</sub> and eventually, MSEs were computed between the
Z<sub>n</sub> and Z’<sub>n</sub> (output dataset) and plotted against
Epochs accordingly.

The obtained MSEs are rescaled by dividing them with the average norms
of the input dataset and Train\_rMSE and Test\_rMSE are calculated.
These are also plotted against Epochs as shown below.

The computational time for this model was 324.51 sec.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture19.png)

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture20.png)

The above plots show the MSE curves of the training and testing
datasets, after epoch 2, both the training and testing MSE curves are
staggered, this narrows the best epoch to stop the autoencoder m<sup>*</sup> = 2.

The corresponding weights and thresholds for this epoch are now fixed
and utilized in the next step.

Step 2: Developing MLP with Auto-Encoder

The weights and thresholds were obtained for the first half of the
L<sub>1</sub> to L<sub>2</sub> autoencoder. Z<sub>n</sub> computed from
X<sub>n</sub> using the weights and thresholds of
MLP<sub>high</sub> are inputs to L<sub>1</sub>.

The weights and thresholds of autoencoder L<sub>1</sub> to L<sub>2</sub>
transform Z<sub>n</sub> into a new input vector K<sub>n</sub> on
L<sub>2</sub>.

_K<sub>n</sub> = RELU(Z<sub>n</sub> * W + B)_

Assumptions:

1.  Batch Size = 50

2.  Epochs = 100

The batch size was reduced to 50 and Epochs were increased to 100 to
optimize accuracies. K<sub>n</sub> was sent as an input to layer
L<sub>2</sub> with the number of perceptrons equal to dim
(K<sub>n</sub>). The MLP was then launched with only 1 hidden layer
H<sub>3</sub> (h<sub>long</sub> = computed using PCA analysis on
K<sub>n</sub> as shown below) and probabilities were predicted. The
train and test AVCREs were also computed at each Epoch and plotted
against Epochs as shown below.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture21.png)

95% PEV = 9, so h<sub>long</sub> = 9

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture22.png)

The AVCREs for the train set computed from this model fall gradually and
reach a plateau whereas the AVCREs for the test set follow a similar
trend in the beginning but start oscillating vigorously after 20 Epochs
till the end.

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture23.png)

Confusion matrix for the train set

From the confusion matrix above, the MLP<sub>long</sub>
predicted classes on the train set with accuracies ranging from 54% to
86%. Class 1 was most misclassified with an accuracy of only 54% while
classes 4 and 5 were classified with the highest accuracies of 84% and
86% respectively. Overall, this model predicted classes with lower
accuracies compared to previous models.

| Class | Percentage of Correct Classification |
| --------- | ---------------------------------------- |
| 1         | 54                                       |
| 2         | 69                                       |
| 3         | 69                                       |
| 4         | 84                                       |
| 5         | 86                                       |
| 6         | 70                                       |
| 7         | 75                                       |

![](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/images/Picture24.png)

Confusion matrix for test set

The MLP<sub>long</sub> also predicted a similar level of
accuracies on the test set ranging from 54 to 86%. Class ‘1’ was most
misclassified while classes 4 and 5 were least misclassified. For class
‘1’, 33% of the observations were misclassified as class 7, and 13%
were misclassified as class 2.

| Class | Percentage of Correct Classification |
| --------- | ---------------------------------------- |
| 1         | 54                                       |
| 2         | 69                                       |
| 3         | 68                                       |
| 4         | 84                                       |
| 5         | 86                                       |
| 6         | 69                                       |
| 7         | 75                                       |

Computation time for this model: 816.82 sec


#### Comparison of performances among different MLP models:

<table>
<thead>
<tr class="header">
<th></th>
<th colspan="2"><br />ML<em>P</em><sub>low</sub></span><br /></th>
<th colspan="2"><br />ML<em>P</em><sub>high</sub></span><br /></th>
<th colspan="2"><br />ML<em>P</em><sub>long</sub></span><br /></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>Epochs, Batch size</td>
<th colspan="2">50, 100</th>
<th colspan="2">50, 100</th>
<th colspan="2">100, 50</th>
</tr>
<tr class="even">
<td>Classes</td>
<td>Train Set</td>
<td>Test Set</td>
<td>Train Set</td>
<td>Test Set</td>
<td>Train Set</td>
<td>Test Set</td>
</tr>
<tr class="odd">
<td>1</td>
<td>66</td>
<td>65</td>
<td>85</td>
<td>84</td>
<td>54</td>
<td>54</td>
</tr>
<tr class="even">
<td>2</td>
<td>87</td>
<td>86</td>
<td>91</td>
<td>89</td>
<td>69</td>
<td>69</td>
</tr>
<tr class="odd">
<td>3</td>
<td>88</td>
<td>87</td>
<td>92</td>
<td>91</td>
<td>69</td>
<td>68</td>
</tr>
<tr class="even">
<td>4</td>
<td>94</td>
<td>94</td>
<td>91</td>
<td>91</td>
<td>84</td>
<td>84</td>
</tr>
<tr class="odd">
<td>5</td>
<td>95</td>
<td>95</td>
<td>95</td>
<td>94</td>
<td>86</td>
<td>86</td>
</tr>
<tr class="even">
<td>6</td>
<td>84</td>
<td>83</td>
<td>92</td>
<td>90</td>
<td>70</td>
<td>69</td>
</tr>
<tr class="odd">
<td>7</td>
<td>85</td>
<td>85</td>
<td>86</td>
<td>86</td>
<td>75</td>
<td>75</td>
</tr>
<tr class="even">
<td><p>Computational Time</p>
<p>(In Seconds)</p></td>
<th colspan="2">262.61</th>
<th colspan="2">324.51</th>
<th colspan="2">816.82</th>
</tr>
</tbody>
</table>

From the table above, the performance improved from MLP<sub>low</sub> to
MLP<sub>high</sub>. However, the MLP<sub>long</sub> showed weak
performance compared to the previous models despite optimizing the batch
size and epochs. MLP<sub>high</sub> utilized higher computational time
when compared to MLPlow for the same batch size 100 and number of epochs
50, but this is not significant as the performance improved. The
MLP<sub>long</sub> took over 13 minutes for a batch size of 50 and a
number of epochs of 100.

# [Source Code](https://github.com/datta-mnv/ForestCovertypeNN/blob/main/Forest_Covertype_NN.ipynb)

### References:

1.  Blackard, Jock A. and Denis J. Dean. 2000. "Comparative Accuracies
    of Artificial Neural Networks and Discriminant Analysis in
    Predicting Forest Cover Types from Cartographic Variables."
    Com*puters and Electronics in Agriculture* 24(3):131-151
