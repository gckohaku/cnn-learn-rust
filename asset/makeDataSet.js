const CHANNEL_VALUE = 3, WIDTH = 5, HEIGHT = 5;

function makeData() {
	const data = [];
	for (let channel = 0; channel < CHANNEL_VALUE; channel++) {
		const cData = [];
		for (let i = 0; i < HEIGHT; i++) {
			const row = [];
			for (let j = 0; j < WIDTH; j++) {
				row.push(Math.random());
			}
			cData.push(row);
		}
		data.push(cData);
	}

	return data;
}

/**
 * 
 * @param {number[][][]} data 
 */
function appendLabel(data) {
	console.log("append");
	const R_INDEX = 0, G_INDEX = 1, B_INDEX = 2;
	const count = [0, 0, 0];
	const pixelsSum = [
		data[R_INDEX].flat().reduce((sum, current) => sum + current, 0),
		data[G_INDEX].flat().reduce((sum, current) => sum + current, 0),
		data[B_INDEX].flat().reduce((sum, current) => sum + current, 0)
	];

	for (let i = 0; i < HEIGHT; i++) {
		for (let j = 0; j < WIDTH; j++) {
			const currentPixel = [data[R_INDEX][i][j], data[G_INDEX][i][j], data[B_INDEX][i][j]]
			const maxIndex = currentPixel.indexOf(Math.max(currentPixel));

			count[maxIndex] += 1;
		}
	}

	let maxCount = Math.max(count);
	let mostStrongIndex = -1;

	console.log(count.filter(v => v === maxCount).length);
	console.log(maxCount);
	console.log(count);

	if (count.filter(v => v === maxCount).length > 1) {
		let currentMaxValue = -1;
		for (let channel = 0; channel < CHANNEL_VALUE; channel++) {
			console.log("test");
			if (currentMaxValue < pixelsSum[channel]) {
				mostStrongIndex = channel;
				currentMaxValue = pixelsSum;
			}
		}
	}
	else {
		console.log("less");
		mostStrongIndex = count.indexOf(maxCount);
	}

	if (mostStrongIndex < 0) {
		throw new Error("'mostStrongIndex' is not to set value");
	}

	let data_label = [0.0, 0.0, 0.0];
	data_label[mostStrongIndex] = 1.0;

	return {
		image: data,
		label: data_label,
	}
}

let data = makeData();
let d = appendLabel(data);

d