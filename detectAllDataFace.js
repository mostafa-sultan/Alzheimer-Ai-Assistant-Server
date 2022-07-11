
const path = require("path");
const tf = require("@tensorflow/tfjs-node");
const canvas = require("canvas");
const faceapi = require("@vladmandic/face-api/dist/face-api.node.js");

const modelPathRoot = "./models";


const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const fs = require('fs');


//mode face detection
let optionsSSDMobileNet;

//function prepare image file
async function image(file) {
  const decoded = tf.node.decodeImage(file);
  const casted = decoded.toFloat();
  const result = casted.expandDims(0);
  decoded.dispose();
  casted.dispose();
  return result;
}


//main function
async function lod() {
  console.log("FaceAPI single-process test");
  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready();
  console.log("Loading FaceAPI models");
  const modelPath = path.join(__dirname, modelPathRoot);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath)
  optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({
    minConfidence: 0.5,
  });
}


//main function
let labeledFaceDescriptors;
async function main(file, filename) {
  // console.log("1");
  if (!labeledFaceDescriptors) {
    // console.log("2");

    await lod()
    labeledFaceDescriptors = await loadLabeledImages();
    // let data = JSON.stringify(labeledFaceDescriptors);
    // fs.writeFileSync('student-2.json', labeledFaceDescriptors);
    // let student = 
    // console.log("3");

    console.log(labeledFaceDescriptors);
  }

  var facename = await faceRecognition(file);
  return [facename];
}

async function faceRecognition(imageFile) {
  // console.log("4");

  const referenceImage = await canvas.loadImage(imageFile)
  const faceMatcher = new faceapi.FaceMatcher( 
    labeledFaceDescriptors,
    0.6
  );
  const displaySize = { width: referenceImage.width, height: referenceImage.height };
  const detections = await faceapi
    .detectSingleFace(referenceImage)
    .withFaceLandmarks()
    .withFaceDescriptor();
  const resizedDetections = faceapi.resizeResults(
    detections,
    displaySize
  );
  // console.log(resizedDetections);
  const results = faceMatcher.findBestMatch(resizedDetections.descriptor)
  return results;
}

 
// function handel traninig image for tran  model
async function loadLabeledImages() {
  // console.log("5");

  const labels = [
    'Mostafa soltan',
    'Mohamed',
    'Hamza',
    'Ali',
    'Omer',
    'Salama',
    'Fady adalat'
  ];
  return Promise.all(
     
    labels.map(async (label) => {
      // console.log("6");
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        
        const img = await canvas.loadImage(
          `http://localhost:3000/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

module.exports = {
  detect: main,
};

