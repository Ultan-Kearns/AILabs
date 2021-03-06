package ai;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class EncogZooRunner {

	public void go() throws Exception {
		//1. topology
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, 16));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 5));

		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 7));
		network.getStructure().finalizeStructure();
		network.reset();
		//2. training data set
		MLDataSet trainingSet = new BasicMLDataSet(data, expected);
		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		double minError = 0.04;
		//3. train
		int epoch = 1;
		System.out.println("INFO TRAINING..........");
		do {
			train.iteration();
			epoch++;
		} while (train.getError() > minError);
		train.finishTraining();
		System.out.println("INFO TRAINING COMPLETE in " + epoch + " epochs with error rate " + train.getError());
		//4. test
		double correct = 0;
		double total = 0;
		for (MLDataPair pair : trainingSet) {
			total++;
			MLData output = network.compute(pair.getInput());
			//compare actual to ideal
			int y = (int) Math.round(output.getData(0));
			int yd = (int) pair.getIdeal().getData(0);
			if(y == yd) {
				correct++;
			}
			System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1) + ", Y="
					+ (int) Math.round(output.getData(0)) + ", Yd=" + (int) pair.getIdeal().getData(0));
		}
		System.out.println("INFO Testing complete ACC= " + (correct / total) * 100 + " %");
	
	}

	private double[][] data = { { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 0, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 0, 0, 1 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 1, 1 },
			{ 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 0, 1, 0 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 1, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.5, 0, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.75, 0, 0, 0 },
			{ 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1 }, { 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 1, 0 },
			{ 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 1 },
			{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.75, 0, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0.5, 0, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0.5, 0, 0, 0 }, { 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0.25, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.25, 0, 1, 1 },
			{ 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.75, 0, 0, 0 }, { 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 1, 1 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.25, 0, 0, 1 },
			{ 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 1, 0 }, { 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 0 },
			{ 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
			{ 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0.75, 0, 1, 0 },
			{ 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.75, 0, 0, 0 },
			{ 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 },
			{ 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0.75, 0, 0, 0 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.75, 0, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 0 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.75, 0, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 1 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 1, 0 },
			{ 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 1 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1 },
			{ 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0 },
			{ 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 1, 1 }, { 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 1, 1 },
			{ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 1, 1 },
			{ 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 1 }, { 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0 }, { 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1 },
			{ 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0.25, 1, 0, 1 }, { 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 },
			{ 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 }, { 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0 },
			{ 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.25, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0.625, 0, 0, 0 }, { 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1 },
			{ 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 1 },
			{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.75, 0, 0, 0 }, { 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0.5, 0, 0, 0 },
			{ 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0.5, 1, 0, 1 }, { 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 0 },
			{ 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1 }, { 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0.25, 1, 0, 0 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.5, 1, 0, 0 }, { 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0.25, 1, 0, 1 },
			{ 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.25, 1, 0, 1 },
			{ 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0.75, 0, 0, 0 }, { 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0.5, 1, 0, 1 },
			{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0.25, 1, 0, 0 } };

	private double[][] expected = { { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 0, 1 },
			{ 0, 0, 0, 0, 0, 0, 1 }, { 0, 1, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 1, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 },
			{ 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 1, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 1, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 }, { 0, 1, 0, 0, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 1, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 1, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 },
			{ 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 1, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0 }, { 1, 0, 0, 0, 0, 0, 0 },
			{ 0, 0, 0, 0, 0, 0, 1 }, { 0, 1, 0, 0, 0, 0, 0 } };

	public static void main(String[] args) throws Exception {
		new EncogZooRunner().go();
	}
}
