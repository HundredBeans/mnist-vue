<template>
  <div>
    <h1>Testing the Model</h1>
    <div v-if="model === null">
      <h2>There is no Model</h2>
      <button @click="loadModel">Load Model</button>
    </div>
    <div v-else>
      <h3>Predicted Number {{ predictNumber }}</h3>
      <VueSignaturePad
        id="signature-pad"
        height="250px"
        width="250px"
        style="border-style: solid; margin: auto"
        ref="signaturePad"
        :options="{ onBegin, onEnd, penColor: 'white', backgroundColor: 'black', dotSize: 10, minWidth: 10, maxWidth: 12 }"
      />
      <div>
        <button @click="save">Save</button>
        <button @click="undo">Undo</button>
      </div>
    </div>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";

export default {
  data: () => {
    return {
      model: null,
      predictNumber: null,
      result: [],
    };
  },
  methods: {
    undo() {
      this.$refs.signaturePad.undoSignature();
    },
    async save() {
      const { isEmpty, data } = this.$refs.signaturePad.saveSignature();
      console.log(isEmpty);
      console.log(data);
    },
    async loadModel() {
      try {
        this.model = await tf.loadLayersModel("localstorage://mnist-model");
      } catch (error) {
        console.log(error);
        this.model = null;
      }
    },
    async onEnd() {
      let canvasElement = document.getElementById("signature-pad").firstChild;
      let imageData = canvasElement.getContext("2d");
      imageData = imageData.getImageData(
        0,
        0,
        canvasElement.width,
        canvasElement.height
      );
      console.log(imageData);
      // Convert the canvas pixels to a Tensor of the matching shape
      let img = tf.browser.fromPixels(imageData, 1);
      img.print();
      // Resize
      img = tf.image.resizeBilinear(img, [28, 28]);
      img = await img.reshape([1, 28, 28, 1]);
      img = tf.cast(img, "float32");
      // Make and format the predications
      const output = this.model.predict(img);
      console.log(this.model);
      img.dispose();
      // Save predictions on the component
      this.result = Array.from(output.dataSync());
      this.getResult(this.result);
      console.log(this.result);
      console.log("predict", Array.from(output.dataSync()));
    },
    onBegin() {
      console.log("=== End ===");
    },
    getResult(result) {
      let maxNumIdx = 0;
      let maxNum = 0;
      result.forEach((item, index) => {
        if (item > maxNum) {
          maxNum = item;
          maxNumIdx = index;
        }
      });
      this.predictNumber = maxNumIdx;
    },
  },
};
</script>

<style>
</style>