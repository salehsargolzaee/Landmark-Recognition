<h1 align="center">Landmark-Tagging-For-Social-Media</h1>

<p align="center"><i>Let's talk about the project on <a href="https://www.linkedin.com/in/saleh-sargolzaee">LinkedIn</a> !</i></p>
<br>

## About the project
Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

<br/>
If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.



## Sample results

 The images below display some sample outputs of my finished project:
 
 ![Sydney_Harbour_Bridge](images/result2.png)
 ![Trevi_Fountain](images/result4.png)
 ![Death_valley2](images/result0.png)
 ![Gateway_of_India](images/result1.png)

## Getting Started

### Dataset

The landmark images are a subset of the Google Landmarks Dataset v2. It can be downloaded using [this link](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip)
You can find license information for the full dataset [on Kaggel](https://www.kaggle.com/competitions/landmark-recognition-challenge/data)


### Steps 

__Notice:__  please be careful with the versions; if you use newer versions of PyTorch and torchvision, there will probably be some errors. So it's recommended to install packages through the steps below:

1. Clone the repo
   ```sh
   git clone https://github.com/salehsargolzaee/Landmark-Recognition   
   ```
2. Change directory to repo folder
   ```sh
   cd path/to/repo/folder
   ```
3. Download the landmark dataset. Unzip the folder and place it in this project's home directory, at the location `data/landmark_images`.

4. Create an environment with required packages
   ```sh
   conda env create -f environment.yaml
   conda activate landmark-tagging
   ```
- or you can use `pip`:

   ```sh
   pip install -r requirements.txt
   ```
5. Run `jupyter notebook`
    
   ```sh
   jupyter notebook
   ```
6. Open `landmark.ipynb`


## Contact

Saleh Sargolzaee - [LinkedIn](https://www.linkedin.com/in/saleh-sargolzaee) - salehsargolzaee@gmail.com

Project Link: [https://github.com/salehsargolzaee/Landmark-Recognition](https://github.com/salehsargolzaee/Landmark-Recognition)

<p align="right">(<a href="#top">back to top</a>)</p>

## :man_astronaut: Show your support

Give a ⭐️ if you liked the project!

