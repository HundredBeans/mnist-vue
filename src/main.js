import Vue from 'vue'
import App from './App.vue'
import './registerServiceWorker'
import router from './router'
import store from './store'

import VueSignaturePad from 'vue-signature-pad';
import vuetify from './plugins/vuetify';
import Toasted from "vue-toasted";

// Vue Toasted
Vue.use(Toasted, {
  position: 'top-center',
  theme: 'bubble',
  iconPack: 'mdi',
  duration: 2000
});

Vue.use(VueSignaturePad);

Vue.config.productionTip = false

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App)
}).$mount('#app')
