import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Semantic Segmentation", page_icon=":smile:", layout="wide", initial_sidebar_state="expanded")
# ----------- General things
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

lm1 = ["Unet_Model1", "Unet_Model2", "ResNet-38", "ResNet-101"]
lm2 = [0.9111, 0.7866, 0.7840, 0.7760]
lm3 = [0.3124, 0.7456, " ", " "]
lm4 = ["Continuous Learning", "Continuous Learning", "Batch Learning", "Batch Learning"]
lm5 = ["[GitHub](https://github.com/WhiteWolf47/lfx_task1)", "[GitHub](https://github.com/WhiteWolf47/lfx_task1)", "https://github.com/itijyou/ademxapp", "https://github.com/TuSimple/TuSimple-DUC"]

#Dataframe for the table
df_bm = pd.DataFrame(list(zip(lm1, lm2, lm3, lm4, lm5)), columns =['Model', 'Accuracy', 'Loss', 'Learning Paradigm', 'Github Link'])

# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Home Page", "Documentation", "Download", "Benchmarking"])

st.sidebar.markdown("""---""")
st.sidebar.write("[Github](https://github.com/WhiteWolf47/cscapes_semantic_segmentation)")
st.sidebar.title("Dataset Sample")

if page == "Home Page":

    # ----------- Inputs
    st.title("Dataset Overview")
    st.write("Cityscapes is a large-scale database which focuses on semantic understanding of urban street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories (flat surfaces, humans, vehicles, constructions, objects, nature, sky, and void). The dataset consists of around 5000 fine annotated images and 20000 coarse annotated ones. Data was captured in 50 cities during several months, daytimes, and good weather conditions. It was originally recorded as video so the frames were manually selected to have the following features: large number of dynamic objects, varying scene layout, and varying background.")
    
    st.image(["img3.png", "img4.png", "img2.png"], width=450)

    st.title("Lifelong learning algorithm overview")
    st.write("Lifelong Machine Learning or Lifelong Learning (LL) is an advanced machine learning (ML) paradigm that learns continuously, accumulates the knowledge learned in the past, and uses/adapts it to help future learning and problem solving. In the process, the learner becomes more and more knowledgeable and better and better at learning. This continuous learning ability is one of the hallmarks of human intelligence.")
    
    st.header("Some of the LL algorithms are:")
    
    st.subheader("1. MTL net (Multi-task learning with neural network) Caruana")
    st.write("Although MTL net (Multi-task learning with neural network) Caruana is described as a lifelong learning method in , it is actually a batch multi-task learning method. Based on our definition of lifelong learning, they are different learning paradigms In MTL net, instead of building a neural network for each individual task, it constructs a universal neural network for all the tasks. This universal neural network uses the same input layer for input from all tasks and uses one output unit for each task (or class in this case). There is also a shared hidden layer in MTL net that is trained in parallel using Back-Propagation on all the tasks to minimize the error on all the tasks. This shared layer allows features developed for one task to be used by other tasks.")
    
    st.subheader("2. LIFELONG EBNN")
    st.write("This lifelong learning approach is in the context of EBNN (Explanation-Based Neural Network), which again leverages the previous task data (or the support set) to improve learning. Concept learning is the goal of this work, which learns a function f : I → {0, 1} to predict if an object represented by a feature vector x ∈ I belongs to a concept (y = 1) or not (y = 0). In this approach, the system first learns a general distance function, d : I × I → [0, 1], considering all the past data (or the support set) and uses this distance function to share or transfer the knowledge of the past task data to the new task TN+1. Given two input vectors, say x and x' , function d computes the probability of x and x0 being members of the same concept (or class), regardless what the concept is.")
    
    st.subheader("3. LTM: A LIFELONG TOPIC MODEL")
    st.write("Topic modeling has been commonly used to discover topics from document collections. However, unsupervised models can generate many incoherent topics. To address this problem, several knowledge-based topic models have been proposed to incorporate prior domain knowledge from the user. This work advances this research much further and shows that without any user input, we can mine the prior knowledge automatically and dynamically from topics already found from a large number of domains. This paper first proposes a novel method to mine such prior knowledge dynamically in the modeling process, and then a new topic model to use the knowledge to guide the model inference. What is also interesting is that this approach offers a novel lifelong learning algorithm for topic discovery, which exploits the big (past) data and knowledge gained from such data for subsequent modeling. Our experimental results using product reviews from 50 domains demonstrate the effectiveness of the proposed approach.")
    st.write("For a detailed overview of the above algorithms, please refer to this [document](https://www.cs.uic.edu/~liub/lifelong-machine-learning-draft.pdf)")

    #Data Sample Display
    st.title("Dataset Sample")
    st.write("The dataset consists of around 5000 fine annotated images and 20000 coarse annotated ones. Data was captured in 50 cities during several months, daytimes, and good weather conditions. It was originally recorded as video so the frames were manually selected to have the following features: large number of dynamic objects, varying scene layout, and varying background.")
    st.header("Training Sample:")
    st.image(["1.jpg", "2.jpg", "3.jpg"], width=450)
    st.header("Validation Sample:")
    st.image(["val/4.jpg", "val/5.jpg", "val/6.jpg"], width=450)
    st.header("Test Sample:")
    st.image(["test/1test.png", "test/2test.png", "test/3test.png"], width=450)

elif page == "Documentation":
    st.title("Dataset partition description")
    l1 = ["flat", "human", "vehicle", "construction", "object", "nature", "sky", "void"]
    l2 = ["road · sidewalk · parking+ · rail track+", "person* · rider*", "car* · truck* · bus* · on rails* · motorcycle* · bicycle* · caravan*+ · trailer*+", "building · wall · fence · guard rail+ · bridge+ · tunnel+", "pole · pole group+ · traffic sign · traffic light", "vegetation · terrain", "sky", "ground+ · dynamic+ · static+"]
    df = pd.DataFrame(list(zip(l1, l2)), columns =['Group', 'Classes'])
    df.reset_index(drop=True, inplace=True)
    st.table(df)

    st.title("Data statistics")
    st.header("Features")
    st.subheader("Polygonal annotations")
    st.write("Dense semantic segmentation")
    st.write("Instance segmentation for vehicle and people")

    st.subheader("Complexity")
    st.write("30 classes")
    st.write("See Class Definitions for a list of all classes and have a look at the applied labeling policy.")

    st.subheader("Diversity")
    st.write("50 cities")
    st.write("Several months (spring, summer, fall)")
    st.write("Daytime")
    st.write("Good/medium weather conditions")
    st.write("5000 annotated images with fine annotations (examples)")
    st.write("20000 annotated images with coarse annotations (examples)")
    st.subheader("MetaData")
    st.write("Preceding and trailing video frames. Each annotated image is the 20th image from a 30 frame video snippets (1.8s)")
    st.write("Corresponding right stereo views")
    st.write("Good/medium weather conditions")
    st.write("Ego-motion data from vehicle odometry")
    st.write("Outside temperature from vehicle sensor")
    st.write("GPS coordinates")
    st.subheader("Extensions by other researchers")
    st.write("Bounding box annotations of people")
    st.write("Images augmented with fog and rain")
    st.subheader("Benchmark suite and evaluation server")
    st.write("Pixel-level semantic labeling")
    st.write("Instance-level semantic labeling")
    st.write("Panoptic semantic labeling")

    st.title("Data Format")
    st.write("The data is provided in the form of images and annotations. The images are provided in the form of png files. The annotations are provided in the form of json files. The json files contain the following fields:")
    st.write("imagePath: The path to the image file.")
    st.write("imageData: The image data in base64 encoding.")
    st.write("imageHeight: The height of the image.")
    st.write("imageWidth: The width of the image.")
    st.write("imageId: The id of the image.")
    st.write("annotations: The annotations of the image.")
    st.write("The annotations field contains a list of annotations. Each annotation contains the following fields:")
    st.write("segmentation: The segmentation mask of the object.")
    st.write("area: The area of the object.")
    st.write("iscrowd: Whether the object is a crowd.")
    st.write("imageId: The id of the image.")
    st.write("bbox: The bounding box of the object.")
    st.write("category_id: The id of the category.")
    st.write("id: The id of the annotation.")
    st.write("categories: The categories of the dataset.")
    st.write("The categories field contains a list of categories. Each category contains the following fields:")
    st.write("supercategory: The supercategory of the category.")
    st.write("id: The id of the category.")
    st.write("name: The name of the category.")
    st.write("Example: /cityscapes/train/leftImg8bit/cityscapes_real/aachen/aachen_000000_000019_leftImg8bit.png")

    st.title("Annotations")
    st.header("Fine annotations")
    st.write("Below are examples of our high quality dense pixel annotations that we provide for a volume of 5000 images. Overlayed colors encode semantic classes (see class definitions). Note that single instances of traffic participants are annotated individually.")
    st.image(["fan1.png", "fan2.png", "fan3.png"], width=450)

    st.header("Coarse annotations")
    st.write("In addition to the fine annotations, we provide coarser polygonal annotations for a set of 20 000 images in collaboration with Pallas Ludens. Again, overlayed colors encode the semantic classes (see class definitions). Note that we do not aim to annotated single instances, however, we marked polygons covering individual objects as such.")
    st.image(["can1.png", "can2.png", "can3.png"], width=450)

    st.header("Videos")
    st.write("The videos below provide further examples of the Cityscapes Dataset. The first video contains roughly 1000 images with high quality annotations overlayed. The second video visualizes the precomputed depth maps using the corresponding right stereo views. The last video is extracted from a long video recording and visualizes the GPS positions as part of the dataset’s metadata. Note that the images are blurred for privacy reasons.")
    
    v1 = open('v1.mp4', 'rb')
    v1b = v1.read()
    st.video(v1b)

elif page == "Download":

    st.title("Instructions")
    st.write("For downloading the data, you need to register on the cityscapes website. After registration, you will receive an email with a link to activate your account. After activation, you can log in and download the data. The data is available in the form of images and annotations. The images are provided in the form of png files. The annotations are provided in the form of json files. The json files contain the following fields:")

    st.title("Download")
    st.write("The dataset is available for download at the following link: https://www.cityscapes-dataset.com/downloads/")

else:
    #Benchmarking
    st.title("Benchmarking")

    st.subheader("The following graph shows the benchmarking results for the different models:")
    df1 = df_bm[::-1]
    fig = px.line(df1, x="Model", y="Accuracy")
    st.plotly_chart(fig)


    st.subheader("The following table shows the benchmarking results for the different models:")
    st.table(df_bm)






