<template>
  <div :class="isVisorOpen ? 'visorOpen' : ''">
    <h1>MNIST Model Training</h1>
    <h2 v-if="isLoading">Downloading the Datasets .......</h2>
    <div v-else>
      <button @click="showExamples(data)">Show Data Examples</button>
      <button @click="openVisor">Open Visor</button>
      <button @click="showModelSummary">Compile Model</button>
      <button @click="train(model, data)">Train the Model</button>
      <button @click="saveModel">Save Model</button>
      <button @click="showEvaluation">Show Evaluation</button>
    </div>
  </div>
</template>

<script>
import { MnistData } from "../dataset/data.js";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_CHANNELS = 1;
const NUM_OUTPUT_CLASSES = 10;

export default {
  name: "Home",
  data: () => {
    return {
      isLoading: false,
      data: new MnistData(),
      model: tf.sequential(),
      visor: tfvis.visor(),
    };
  },
  computed: {
    isVisorOpen() {
      return this.visor.isOpen();
    },
  },
  async created() {
    this.isLoading = true;
    await this.data.load();
    this.isLoading = false;
  },
  methods: {
    async showExamples(data) {
      await this.visor.open();
      // Create a container in the visor
      const surface = tfvis
        .visor()
        .surface({ name: "Input Data Examples", tab: "Input Data" });

      // Get the examples
      const examples = data.nextTestBatch(20);
      const numExamples = examples.xs.shape[0];

      // Create a canvas element to render each example
      for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
          // Reshape the image to 28x28 px
          return examples.xs
            .slice([i, 0], [1, examples.xs.shape[1]])
            .reshape([28, 28, 1]);
        });

        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = "margin: 4px;";
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
      }
    },
    createModel() {
      // Define the first CNN layer. Using conv2d layer,
      // with inputShape as the shape of data that first enter this layer
      // KernelSize is the size of the sliding convolutional filter window (n x n)
      // activation type usualy use relu or sigmoid
      this.model.add(
        tf.layers.conv2d({
          inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
          kernelSize: 5,
          filters: 8,
          strides: 1,
          activation: "relu",
          kernelInitializer: "varianceScaling",
        })
      );

      // The MaxPooling layer acts as a sort of downsampling using max values
      // in a region instead of averaging.
      this.model.add(
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
      );

      // Repeat another conv2d + maxPooling stack.
      // Add more filters in the convolution.
      this.model.add(
        tf.layers.conv2d({
          kernelSize: 5,
          filters: 16,
          strides: 1,
          activation: "relu",
          kernelInitializer: "varianceScaling",
        })
      );
      this.model.add(
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] })
      );

      // Now we flatten the output from the 2D filters into a 1D vector to prepare
      // it for input into our last layer. This is common practice when feeding
      // higher dimensional data to a final classification output layer.
      this.model.add(tf.layers.flatten());

      // Our last layer is a dense layer which has 10 output units, one for each
      // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
      // activation softmax for probability distribution
      this.model.add(
        tf.layers.dense({
          units: NUM_OUTPUT_CLASSES,
          kernelInitializer: "varianceScaling",
          activation: "softmax",
        })
      );
      // Optimizer as implementation of the gradient descent algorithm
      const optimizer = tf.train.adam();
      this.model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      return this.model;
    },
    // Train function
    async train(model, data) {
      const metrics = ["loss", "val_loss", "acc", "val_acc"];
      const container = {
        name: "Model Training",
        tab: "Model Training",
        styles: { height: "1000px" },
      };
      await console.log("testImages length", data.trainImages.length);
      // This callbacks is used to show metrics for each model fit callback
      const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
      const BATCH_SIZE = 512;
      const TRAIN_DATA_SIZE = 12000;
      const TEST_DATA_SIZE = 2000;
      const [
        trainXs,
        trainYs,
        testXs,
        testYs,
      ] = await data.getUnshuffledTrainTestData(
        TRAIN_DATA_SIZE,
        TEST_DATA_SIZE
      );
      // const trainXs = await tf.tensor4d(data.trainImages, [
      //   data.trainImages.length / 784,
      //   IMAGE_WIDTH,
      //   IMAGE_HEIGHT,
      //   IMAGE_CHANNELS,
      // ]);
      // const trainYs = await tf.tensor2d(data.trainLabels, [
      //   data.trainLabels.length / 10,
      //   NUM_OUTPUT_CLASSES,
      // ]);
      // const testXs = await tf.tensor4d(data.testImages, [
      //   data.testImages.length / 784,
      //   IMAGE_WIDTH,
      //   IMAGE_HEIGHT,
      //   IMAGE_CHANNELS,
      // ]);
      // const testYs = await tf.tensor2d(data.testLabels, [
      //   data.testLabels.length / 10,
      //   NUM_OUTPUT_CLASSES,
      // ]);

      // const [trainXs, trainYs] = tf.tidy(() => {
      //   const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      //   return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
      // });

      // const [testXs, testYs] = tf.tidy(() => {
      //   const d = data.nextTestBatch(TEST_DATA_SIZE);
      //   return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
      // });

      return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks,
      });

      // return await model.fit(trainXs, trainYs, {
      //   batchSize: BATCH_SIZE,
      //   validationData: [testXs, testYs],
      //   epochs: 10,
      //   shuffle: true,
      //   callbacks: fitCallbacks,
      // });
    },
    doPrediction(model, data, testDataSize = 1000) {
      const IMAGE_WIDTH = 28;
      const IMAGE_HEIGHT = 28;
      const testData = data.nextTestBatch(testDataSize);
      const testxs = testData.xs.reshape([
        testDataSize,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        1,
      ]);
      const labels = testData.labels.argMax(-1);
      const preds = model.predict(testxs).argMax(-1);
      testxs.dispose();
      return [preds, labels];
    },
    async showAccuracy(model, data) {
      const [preds, labels] = this.doPrediction(model, data);
      const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
      const container = { name: "Accuracy", tab: "Evaluation" };
      tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

      labels.dispose();
    },
    async showConfusion(model, data) {
      const [preds, labels] = this.doPrediction(model, data);
      const confusionMatrix = await tfvis.metrics.confusionMatrix(
        labels,
        preds
      );
      const container = { name: "Confusion Matrix", tab: "Evaluation" };
      tfvis.render.confusionMatrix(
        container,
        { values: confusionMatrix },
        classNames
      );

      labels.dispose();
    },
    async showEvaluation() {
      await this.showAccuracy(this.model, this.data);
      await this.showConfusion(this.model, this.data);
    },
    async openVisor() {
      await this.visor.open();
    },
    async showModelSummary() {
      const model = await this.createModel();
      // Show Model Summary
      tfvis.show.modelSummary(
        { name: "Model Architecture", tab: "Model Summary" },
        model
      );
    },
    async saveModel() {
      await this.model.save("localstorage://mnist-model");
    },
  },
  destroyed() {
    this.visor.close();
  },
};
</script>
<style lang="scss" scoped>
.visorOpen {
  float: left;
}
</style>