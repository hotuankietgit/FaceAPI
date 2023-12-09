const express = require("express");
const app = express();
const fileUpload = require("express-fileupload");
const cors = require("cors");
const fs = require("fs");
const tf = require('@tensorflow/tfjs-node');
// const faceapi = require("./face-api.js");
const faceapi = require("@vladmandic/face-api")
const canvas = require('canvas')
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })


async function LoadModels() {
    await faceapi.nets.faceRecognitionNet.loadFromDisk(__dirname + "/models");
    await faceapi.nets.faceLandmark68Net.loadFromDisk(__dirname + "/models");
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(__dirname + "/models");
}

LoadModels();

app.use(cors())
app.use(fileUpload())

app.get("/", (req,res) => {
    res.send("Hello");
})

app.post("/upload", async (req,res) => {
    if (req.body.week == 1){
        let sampleFile1 = req.files.file1;
        let sampleFile2 = req.files.file2;
        let sampleFile3 = req.files.file3;
        
        let folderUpload = __dirname + "/uploads/" + req.body.studentID + "_" + req.body.classID;
        fs.mkdirSync(folderUpload)
        let uploadPath1 = folderUpload + "/" + "1.jpg";
        let uploadPath2 = folderUpload + "/" + "2.jpg";
        let uploadPath3 = folderUpload + "/" + "3.jpg";
        sampleFile1.mv(uploadPath1, function(err) {
            if (err)
              return res.status(500).send(err);
        });

        sampleFile2.mv(uploadPath2, function(err) {
            if (err)
              return res.status(500).send(err);
        });

        sampleFile3.mv(uploadPath3, function(err) {
            if (err)
              return res.status(500).send(err);
        });

        return res.send("Success");
    }else if (req.body.week > 1){
        let sampleFile = req.files.file;
        //console.log(sampleFile);

        
        const labels = ["1","2","3"]
        
        let img = await canvas.loadImage(req.files.file.data)
        
        //console.log(img)
        let faceDescriptions = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors()
        faceDescriptions = faceapi.resizeResults(faceDescriptions, img)
        const labeledFaceDescriptors = await Promise.all(
            labels.map(async label => {
                const imgURL = `./uploads/520H0380_555666/${label}.jpg`;
                const fileBuffer = fs.readFileSync(imgURL);
                const img = await canvas.loadImage(fileBuffer);

                const faceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

                if (!faceDescription) {
                    throw new Error("no faces detected for ${label}")
                }
                const faceDescriptors = [faceDescription.descriptor]
                return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
            })
        )

        const threshold = 0.6
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, threshold)
        const results = faceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor))


        return res.json(results);
    }
})

app.listen(8080, () => console.log("Server is running"))