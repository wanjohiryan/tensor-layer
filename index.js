// forgive the naming of my variables :)

const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');

const app = express();


app.use(cors());
app.use(bodyParser.json());

const imageUrl = "https://i.redd.it/v3is7lp82ty81.jpg"

app.get("/test", (req, res) => {
    res.send("Hello World 🕵🏽‍♀️🕵🏽‍♀️😎😎");
});

app.get("/", async (req, res) => {
    try {
        const model = await tf.loadGraphModel(
            'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_large_100_224/feature_vector/5/default/1',
            { fromTFHub: true });

        const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });

        const imageTensor = tf.node.decodeImage(response.data, 3);

        const processedImage = preprocess(imageTensor);

        const predictions = await model.predict(processedImage);

        const pred = await  predictions.data();

        const responseData = JSON.stringify(predictions);

        console.log(responseData);

        res.send(pred); // Send the predictions back to the client.
    } catch (error) {
        console.error(error);

        res.send(error);
    }
});


const preprocess = (imageTensor) => {
    const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0];
    let squareCrop;
    if (widthToHeight > 1) {
        const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1];
        const cropTop = (1 - heightToWidth) / 2;
        const cropBottom = 1 - cropTop;
        squareCrop = [[cropTop, 0, cropBottom, 1]];
    } else {
        const cropLeft = (1 - widthToHeight) / 2;
        const cropRight = 1 - cropLeft;
        squareCrop = [[0, cropLeft, 1, cropRight]];
    }
    // Expand image input dimensions to add a batch dimension of size 1.
    const crop = tf.image.cropAndResize(
        tf.expandDims(imageTensor), squareCrop, [0], [224, 224]);
    return crop.div(255);
};

const port = process.env.PORT || 8080;
app.listen(port, () => {
    console.log(`Listening on port ${port}`);
});