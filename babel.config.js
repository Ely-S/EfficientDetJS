module.exports = {
  presets: [
    ['@babel/preset-env', {
      targets: "> 2%",
      loose: true,
      "modules": false
    }],

    '@babel/preset-typescript',
  ],
};

