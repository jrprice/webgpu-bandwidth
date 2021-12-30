const path = require('path');

module.exports = {
  entry: './src/benchmark.js',
  mode: 'development',
  output: {
    filename: 'benchmark.js',
    path: path.resolve(__dirname, 'dist'),
  },
  module: {
    rules: [
      {
        test: /\.wgsl$/i,
        use: 'raw-loader',
      },
    ],
  },
};
