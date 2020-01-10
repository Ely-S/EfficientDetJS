module.exports = {
  roots: ['.'],
  transform: {
    '^.+\\.tsx?$': 'ts-jest',
  },
  transformIgnorePatterns: [],
  testRegex: '(/__tests__/.*|(\\.|/)(test|spec))\\.tsx?$',

  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],

globals: {
  'ts-jest': {
    isolatedModules: true
  },
}

}

