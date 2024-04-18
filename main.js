let model;

async function learnLinear() {
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
    const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);


    const surface = { name: 'Loss', tab: 'Training' };
    const history = [];

    await model.fit(xs, ys, {
            epochs: 300, 
            callbacks : {
                onEpochEnd: (epoch, logs) => {
                    history.push(logs);
                    tfvis.show.history(surface, history, ['loss']);
                }
            }
         },  
    );

    console.log("Entrenado");
    alert("Modelo entrenado. Listo para usarse");
}

document.getElementById('predecir').addEventListener('click', () => {
    if (!model) {
        alert("No se ha inicializado el modelo");
        console.error("Modelo no inicializado");
        return;
    }

    const input_number = parseFloat(document.getElementById('input_number').value);
    if (isNaN(input_number)) {
        console.error("Valor de entrada vacío");
        alert("Valor de entrada vacío");
        return;
    }

    const prediction = model.predict(tf.tensor2d([input_number], [1, 1]));
    const salida = `El resultado de predecir ${input_number} es ${prediction.dataSync()[0]}`;
    document.getElementById('output_field').innerHTML = salida;
});
