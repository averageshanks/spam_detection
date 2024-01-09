import streamlit as st
from utils import load_lottie
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Resource", page_icon=":ninja:", layout="wide")

content, icon = st.columns([3, 1])

with st.container():
    with content:
        st.title("ALGORITHMS")
        st.subheader("Logistic Regression")
        st.write(
            """Logistic regression is a calculation used to predict a binary outcome: either something happens, or does not. This can be exhibited as Yes/No, Pass/Fail, Alive/Dead, etc. 

Independent variables are analyzed to determine the binary outcome with the results falling into one of two categories. The independent variables can be categorical or numeric, but the dependent variable is always categorical. Written like this:

            P(Y=1|X) or P(Y=0|X)

It calculates the probability of dependent variable Y, given independent variable X. 

This can be used to calculate the probability of a word having a positive or negative connotation (0, 1, or on a scale between). Or it can be used to determine the object contained in a photo (tree, flower, grass, etc.), with each object given a probability between 0 and 1."""
        )

    with icon:
        lottie_animation = load_lottie("./animation/ai.json")
        st_lottie(lottie_animation, height=300, width=300, key="ai", speed=2)

with st.container():
    st.write("---")
    content, ani = st.columns([4, 2])
    with content:
        st.subheader("Naive Bayes")
        st.write(
            """Naive Bayes calculates the possibility of whether a data point belongs within a certain category or does not. In text analysis, it can be used to categorize words or phrases as belonging to a preset “tag” (classification) or not. For example:"""
        )
        st.image("./animation/table.webp", width=600)

        st.write(
            """To decide whether or not a phrase should be tagged as “sports,” you need to calculate:"""
        )
        st.image("./animation/formula.webp", width=400)
        st.write(
            """Or… the probability of A, if B is true, is equal to the probability of B, if A is true, times the probability of A being true, divided by the probability of B being true."""
        )
    with ani:
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        lottie_animation = load_lottie("./animation/algorithm.json")
        st_lottie(lottie_animation, height=350, width=350, key="algorithm")

with st.container():
    st.write("---")
    st.subheader("Bagging Classifier")
    st.write(
        """Bagging (or Bootstrap aggregating) is a type of ensemble learning in which multiple base models are trained independently in parallel on different subsets of the training data. Each subset is generated using bootstrap sampling, in which data points are picked at random with replacement. In the case of the Bagging classifier, the final prediction is made by aggregating the predictions of the all-base model, using majority voting. In the case of regression, the final prediction is made by averaging the predictions of the all-base model, and that is known as bagging regression.

    """
    )

    st.image("./animation/bagging-classifier.png")

    st.write("---")
    st.subheader("Decision Tree")
    st.write(
        """A decision tree is a supervised learning algorithm that is perfect for classification problems, as it’s able to order classes on a precise level. It works like a flow chart, separating data points into two similar categories at a time from the “tree trunk” to “branches,” to “leaves,” where the categories become more finitely similar. This creates categories within categories, allowing for organic classification with limited human supervision.

To continue with the sports example, this is how the decision tree works:"""
    )

    st.image("./animation/decision-tree-sports.webp")

    st.write("---")
    st.subheader("Random Forest")
    st.write(
        """The random forest algorithm is an expansion of decision tree, in that you first construct a multitude of decision trees with training data, then fit your new data within one of the trees as a “random forest.”

It, essentially, averages your data to connect it to the nearest tree on the data scale. Random forest models are helpful as they remedy for the decision tree’s problem of “forcing” data points within a category unnecessarily. """
    )
with st.container():
    st.write("---")
    st.subheader("MLP (Multi-layer Perceptron )")
    st.write(
        """Multi-layer perception is also known as MLP. It is fully connected dense layers, which transform any input dimension to the desired dimension. A multi-layer perception is a neural network that has multiple layers. To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other neurons.

A gentle introduction to neural networks and TensorFlow can be found here:

Neural Networks
Introduction to TensorFlow
A multi-layer perceptron has one input layer and for each input, there is one neuron(or node), it has one output layer with a single node for each output and it can have any number of hidden layers and each hidden layer can have any number of nodes. A schematic diagram of a Multi-Layer Perceptron (MLP) is depicted below.

    """
    )

    st.image("./animation/nodeNeural.jpg")

    st.write("---")
    st.subheader("Decision Tree")
    st.write(
        """A decision tree is a supervised learning algorithm that is perfect for classification problems, as it’s able to order classes on a precise level. It works like a flow chart, separating data points into two similar categories at a time from the “tree trunk” to “branches,” to “leaves,” where the categories become more finitely similar. This creates categories within categories, allowing for organic classification with limited human supervision.

To continue with the sports example, this is how the decision tree works:"""
    )

    st.image("./animation/decision-tree-sports.webp")

    st.write("---")
    st.subheader("Random Forest")
    st.write(
        """The random forest algorithm is an expansion of decision tree, in that you first construct a multitude of decision trees with training data, then fit your new data within one of the trees as a “random forest.”

It, essentially, averages your data to connect it to the nearest tree on the data scale. Random forest models are helpful as they remedy for the decision tree’s problem of “forcing” data points within a category unnecessarily. """
    )
    with st.container():
        col1, col2 = st.columns([4, 2])
        col1.image("./animation/random_forest.png")
        with col2:
            lottie_animation = load_lottie("./animation/robot.json")
            st_lottie(lottie_animation, height=350, width=350, key="robot")

    st.write("---")
    st.subheader("Support Vector Machines")
    st.write(
        """A support vector machine (SVM) uses algorithms to train and classify data within degrees of polarity, taking it to a degree beyond X/Y prediction. 

For a simple visual explanation, we’ll use two tags: red and blue, with two data features: X and Y, then train our classifier to output an X/Y coordinate as either red or blue."""
    )

    st.image("./animation/svm-example-1.webp")
    st.write("#")

    st.write(
        """The SVM then assigns a hyperplane that best separates the tags. In two dimensions this is simply a line. Anything on one side of the line is red and anything on the other side is blue. In sentiment analysis, for example, this would be positive and negative.

In order to maximize machine learning, the best hyperplane is the one with the largest distance between each tag:"""
    )

    st.image("./animation/svm-example-2.webp")
    st.write("#")

    st.write(
        """However, as data sets become more complex, it may not be possible to draw a single line to classify the data into two camps:"""
    )

    st.image("./animation/svm-example-3.webp")
    st.write("#")

    st.write(
        """Using SVM, the more complex the data, the more accurate the predictor will become. Imagine the above in three dimensions, with a Z-axis added, so it becomes a circle.

Mapped back to two dimensions with the best hyperplane, it looks like this:"""
    )

    st.image("./animation/svm-example-4.webp")
    st.write("#")
    st.write(
        """SVM allows for more accurate machine learning because it’s multidimensional."""
    )
