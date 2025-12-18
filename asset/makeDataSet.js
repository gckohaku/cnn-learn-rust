const CHANNEL_VALUE = 3,
	WIDTH = 5,
	HEIGHT = 5;

	/**
	 * 
	 * @returns {[number[][][], string]}
	 */
function makeData() {
	const data = [];
	let dataString = "(";
	for (let channel = 0; channel < CHANNEL_VALUE; channel++) {
		const cData = [];
		dataString += "(";
		for (let i = 0; i < HEIGHT; i++) {
			const row = [];
			dataString += "(";
			for (let j = 0; j < WIDTH; j++) {
				row.push(Math.random());
				dataString += `${Math.random()}, `;
			}
			cData.push(row);
			dataString += "), ";
		}
		data.push(cData);
		dataString += "), ";
	}

	dataString += ")";
	return [data, dataString];
}

/**
 *
 * @param {[number[][][], string]} dataArray
 */
function appendLabel(dataArray) {
	const data = dataArray[0];
	let dataString = dataArray[1];

	const R_INDEX = 0,
		G_INDEX = 1,
		B_INDEX = 2;
	const count = [0, 0, 0];
	const pixelsSum = [data[R_INDEX].flat().reduce((sum, current) => sum + current, 0), data[G_INDEX].flat().reduce((sum, current) => sum + current, 0), data[B_INDEX].flat().reduce((sum, current) => sum + current, 0)];

	for (let i = 0; i < HEIGHT; i++) {
		for (let j = 0; j < WIDTH; j++) {
			const currentPixel = [data[R_INDEX][i][j], data[G_INDEX][i][j], data[B_INDEX][i][j]];
			const maxIndex = currentPixel.indexOf(Math.max(...currentPixel));

			count[maxIndex] += 1;
		}
	}

	let maxCount = Math.max(...count);
	let mostStrongIndex = -1;

	if (count.filter((v) => v === maxCount).length > 1) {
		let currentMaxValue = -1;
		for (let channel = 0; channel < CHANNEL_VALUE; channel++) {
			if (currentMaxValue < pixelsSum[channel]) {
				mostStrongIndex = channel;
				currentMaxValue = pixelsSum;
			}
		}
	} else {
		mostStrongIndex = count.indexOf(maxCount);
	}

	if (mostStrongIndex < 0) {
		throw new Error("'mostStrongIndex' is not to set value");
	}

	let dataLabel = [0.0, 0.0, 0.0];
	dataLabel[mostStrongIndex] = 1.0;

	let labelString = "(";
	for (let i = 0; i < CHANNEL_VALUE; i++) {
		labelString += `${dataLabel[i].toFixed(1)}, `
	}
	labelString += "), ";

	return `(image: ${dataString}, label: ${labelString} ),`;
}

/**
 *
 * @param {number} times
 */
function makeDataSet(times) {
	dataset = [];
	let dataString = "(data: (";
	for (let i = 0; i < times; i++) {
		dataString += `${appendLabel(makeData())}`;
		dataString += "\n";
	}

	dataString += "), )";
	return dataString.replace(/\\n/g, "\n");
}