<template>
  <v-row>
    <v-col class="justify-center align-center tagline-feature">
      <v-card width="500" style="margin: auto">
        <v-card-title class="title justify-space-between">
          <h3>MNIST Model Training</h3>
          <v-spacer></v-spacer>
          <v-btn color="black" v-if="!isVisorOpen" @click="openVisor"
            >Open Visor</v-btn
          >
          <v-btn color="black" v-else @click="closeVisor">Close Visor</v-btn>
        </v-card-title>

        <v-window v-model="step">
          <v-window-item :value="1">
            <v-card-text class="font-weight-bold text-justify text--primary">
              In this project, I use
              <a href="http://yann.lecun.com/exdb/mnist/" target="_blank"
                >MNIST</a
              >
              Dataset that has been downloaded previously using a script. You
              can see some of the MNIST data by clicking the button below. Each
              image is 28 x 28 px Size and has 1 color channel as it is a
              greyscale image as you can see in the visor panel. This MNIST
              dataset will be used to train our model so it can predict or
              detect the digits that you drawn in
              <router-link to="/test">this page</router-link>
            </v-card-text>
            <v-btn @click="showExamples">Show Data Example</v-btn>
          </v-window-item>
          <v-window-item :value="2">
            <v-card-text class="font-weight-bold text--primary">
              You can see the model architecture that I use by clicking the
              button below. You can try to save the untrained model and test it using your handdrawn digits in <router-link to="/test">this page</router-link> and see the difference before and after the model has been trained. <br> ** will add some input so user can change the parameters of the model **
            </v-card-text>
            <v-btn @click="showModelSummary">Compile Model</v-btn>
          </v-window-item>
          <v-window-item :value="3">
            <v-card-text class="font-weight-bold text--primary">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean
              nec pulvinar urna. Curabitur malesuada faucibus enim, eget
              molestie metus eleifend non. Cras nibh enim, aliquet quis nulla
              sit amet, viverra scelerisque risus. Morbi nulla sem, consectetur
              non congue ut, tristique vitae arcu. Etiam vestibulum nisi ut
              metus vulputate, non euismod nisi imperdiet. Nunc facilisis
              lobortis egestas. Donec in tincidunt metus, et tincidunt turpis.
              Nullam interdum sagittis volutpat.
            </v-card-text>
            <v-btn @click="train">Train the Model</v-btn>
          </v-window-item>
          <v-window-item :value="4">
            <v-card-text class="font-weight-bold text--primary">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean
              nec pulvinar urna. Curabitur malesuada faucibus enim, eget
              molestie metus eleifend non. Cras nibh enim, aliquet quis nulla
              sit amet, viverra scelerisque risus. Morbi nulla sem, consectetur
              non congue ut, tristique vitae arcu. Etiam vestibulum nisi ut
              metus vulputate, non euismod nisi imperdiet. Nunc facilisis
              lobortis egestas. Donec in tincidunt metus, et tincidunt turpis.
              Nullam interdum sagittis volutpat.
            </v-card-text>
            <v-btn @click="showEvaluation">Show Evaluation</v-btn>
          </v-window-item>
        </v-window>
        <br />
        <v-card-actions>
          <v-btn :disabled="step === 1" text @click="step--">Back</v-btn>
          <v-spacer></v-spacer>
          <v-btn v-if="step > 1" color="black" @click="saveModel"
            >Save Model</v-btn
          >
          <v-spacer></v-spacer>
          <v-btn v-if="step < 4" color="black" @click="step++">Next</v-btn>
          <v-btn v-if="step === 4" color="black" @click="runTest"
            >Run Test</v-btn
          >
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>
</template>

<script>
export default {
  props: {
    openVisor: {
      type: Function,
      required: true,
    },
    closeVisor: {
      type: Function,
      required: true,
    },
    showExamples: {
      type: Function,
      required: true,
    },
    showModelSummary: {
      type: Function,
      required: true,
    },
    train: {
      type: Function,
      required: true,
    },
    showEvaluation: {
      type: Function,
      required: true,
    },
    saveModel: {
      type: Function,
      required: true,
    },
    isVisorOpen: {
      type: Boolean,
      required: true,
    },
  },
  data: () => ({
    step: 1,
  }),
  methods: {
    runTest() {
      this.$router.push("/test");
    },
  },
};
</script>

<style lang="scss" scoped>
.tagline-feature {
  padding: 30px;
}
</style>