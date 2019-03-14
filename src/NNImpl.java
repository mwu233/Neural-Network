///////////////////////////////////////////////////////////////////////////////
// Project:			 Homework 4	
// Main Class File:  DigitClassifier.java
// File:             NNImpl.java
// Semester:         CS540 Section 2 Spring 2018
//
// Author:           Meiliu Wu (mwu233@wisc.edu)
// Lecturer's Name:  Chuck Dyer
///////////////////////////////////////////////////////////////////////////////

import java.util.*;

/**
 * The main class that handles the entire network Has multiple attributes each
 * with its own use
 * 
 * <p>Bugs: Not found.
 * 
 * @author Meiliu Wu
 */

public class NNImpl {
	private ArrayList<Node> inputNodes; // list of the output layer nodes.
	private ArrayList<Node> hiddenNodes; // list of the hidden layer nodes
	private ArrayList<Node> outputNodes; // list of the output layer nodes

	private ArrayList<Instance> trainingSet; // the training set

	private double learningRate; // variable to store the learning rate
	private int maxEpoch; // variable to store the maximum number of epochs
	private Random random; // random number generator to shuffle the training set

	/**
	 * This constructor creates the nodes necessary for the neural network Also
	 * connects the nodes of different layers After calling the constructor the last
	 * node of both inputNodes and hiddenNodes will be bias nodes.
	 */

	NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random,
			Double[][] hiddenWeights, Double[][] outputWeights) {
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;
		this.random = random;

		// input layer nodes
		inputNodes = new ArrayList<>();
		int inputNodeCount = trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();
		for (int i = 0; i < inputNodeCount; i++) {
			Node node = new Node(0);
			inputNodes.add(node);
		}

		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);

		// hidden layer nodes
		hiddenNodes = new ArrayList<>();
		for (int i = 0; i < hiddenNodeCount; i++) {
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for (int j = 0; j < inputNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}

		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);

		// Output node layer
		outputNodes = new ArrayList<>();
		for (int i = 0; i < outputNodeCount; i++) {
			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for (int j = 0; j < hiddenNodes.size(); j++) {
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}
			outputNodes.add(node);
		}
	}

	/**
	 * Get the prediction from the neural network for a single instance - one image
	 * Return the idx with highest output values. For example if the outputs of the
	 * outputNodes are [0.1, 0.5, 0.2], it should return 1. The parameter is a
	 * single instance
	 */

	public int predict(Instance instance) {
		// TODO: add code here

		// calculates the output (i.e., the index of the class) from the neural network
		// for a given example
		// to get the three double output values

		int inputNodesSize = inputNodes.size();
		int hiddenNodesSize = hiddenNodes.size();
		int outputNodesSize = outputNodes.size();
		//////////////////////////////////////////////////////
		// propagate the inputs forward to compute the outputs
		//////////////////////////////////////////////////////

		// for the input layer
		// set the input values for the input nodes
		for (int k = 0; k < inputNodesSize - 1; k++) { // excluding the bias node in the input layer
			// instance's attributes size is inputNodesSize - 1
			inputNodes.get(k).setInput(instance.attributes.get(k));
		}

		// for the hidden layer
		// calculate hidden nodes' output values
		for (int l = 0; l < hiddenNodesSize - 1; l++) { // excluding the bias node in the hidden layer
			hiddenNodes.get(l).setHiddenInput();
			hiddenNodes.get(l).calculateHiddenOutput();
		}

		// for the output layer
		double[] a_o_outputLayer = new double[outputNodesSize];
		double denominator = 0.0;
		// calculate output nodes' output values
		for (int o = 0; o < outputNodesSize; o++) {
			outputNodes.get(o).setOutputInput();
			// calculate the numerator of the Softmax equation
			outputNodes.get(o).calculateOutputNumerator();
			// accumulate the denominator of the Softmax equation
			denominator += Math.pow(Math.E, outputNodes.get(o).getInputValue());
		}

		for (int o = 0; o < outputNodesSize; o++) {
			outputNodes.get(o).calculateOutput_FinalSoftmaxOutput(denominator);
			a_o_outputLayer[o] = outputNodes.get(o).getOutput();
		}

		double max_sofar = a_o_outputLayer[0];
		int max_sofar_index = 0;
		for (int i = 1; i < outputNodesSize; i++) {
			if (max_sofar < a_o_outputLayer[i]) {
				max_sofar = a_o_outputLayer[i];
				max_sofar_index = i;
			}
		}

		// get the index that has the maximum value among the three double output values
		return max_sofar_index;
	}

	/**
	 * Train the neural networks with the given parameters
	 * <p>
	 * The parameters are stored as attributes of this class
	 */

	public void train() {
		// TODO: add code here

		// trains the neural network using:
		// a) a training set,
		// b) fixed learning rate, and
		// c) number of epochs (provided as input to the program).
		// This function also prints the total Cross-Entropy loss on all the training
		// examples after each epoch.

		// get the # of instances in the training set, i.e., the # of images
		int trainsetSize = trainingSet.size();
		int inputNodesSize = inputNodes.size();
		int hiddenNodesSize = hiddenNodes.size();
		int outputNodesSize = outputNodes.size();

		// for each epoch
		for (int i = 0; i < maxEpoch; i++) {

			// shuffle the training set once before every epoch
			Collections.shuffle(trainingSet, random);

			// trainingSet is an ArrayList of instances (images)
			// for each instance/example/image
			for (int j = 0; j < trainsetSize; j++) {
				//////////////////////////////////////////////////////
				// propagate the inputs forward to compute the outputs
				//////////////////////////////////////////////////////

				// for the input layer
				// set the input values for the input nodes
				for (int k = 0; k < inputNodesSize - 1; k++) { // excluding the bias node in the input layer
					// instance's attributes size is inputNodesSize - 1
					inputNodes.get(k).setInput(trainingSet.get(j).attributes.get(k));
				}

				// for the hidden layer
				// calculate hidden nodes' output values
				for (int l = 0; l < hiddenNodesSize - 1; l++) { // excluding the bias node in the hidden layer
					hiddenNodes.get(l).setHiddenInput();
					hiddenNodes.get(l).calculateHiddenOutput();
				}

				// for the output layer
				double denominator = 0.0;
				// calculate output nodes' output values
				for (int o = 0; o < outputNodesSize; o++) {
					outputNodes.get(o).setOutputInput();
					// calculate the numerator of the Softmax equation
					outputNodes.get(o).calculateOutputNumerator();
					// accumulate the denominator of the Softmax equation
					denominator += Math.pow(Math.E, outputNodes.get(o).getInputValue());
				}

				for (int o = 0; o < outputNodesSize; o++) {
					outputNodes.get(o).calculateOutput_FinalSoftmaxOutput(denominator);
				}

				//////////////////////////////////////////////////////
				// propagate deltas backward from output layer to input layer
				//////////////////////////////////////////////////////

				// calculate the delta for each output node
				for (int q = 0; q < outputNodesSize; q++) {
					if (trainingSet.get(j).classValues.get(q).intValue() == 1) {
						outputNodes.get(q).calculateOutput_Target1_Delta();
					} else {
						outputNodes.get(q).calculateOutput_Target0_Delta();
					}
				}

				// for each hidden node in the hidden layer
				// the sum of (delta k * weight from hidden(j) to output(k))
				for (int r = 0; r < hiddenNodesSize - 1; r++) {
					hiddenNodes.get(r).calculateHidden_OutputGradient();
					double sum = 0.0;
					for (int q = 0; q < outputNodesSize; q++) {
						sum += outputNodes.get(q).parents.get(r).weight * outputNodes.get(q).getDelta();
					}
					hiddenNodes.get(r).calculateHidden_Delta(sum);
					// update weights between hidden nodes and input layer nodes
					hiddenNodes.get(r).updateWeight(learningRate);
				}
				// update weights between output nodes and hidden layer nodes
				for (int q = 0; q < outputNodesSize; q++) {
					outputNodes.get(q).updateWeight(learningRate);
				}

			} // end of instances

			// Loss should be calculated once all forward and backward passes are done
			// so the weights of the network are the same in each loss calculation for an instance
			
			//////////////////////////////////////////////////////
			// Accumulate the loss after each example/instance
			//////////////////////////////////////////////////////
			double total_loss = 0.0;
			for (int j = 0; j < trainsetSize; j++) {
				total_loss += loss(trainingSet.get(j));
			}

			// print the total Cross-Entropy loss after each epoch
			System.out.format("Epoch: " + i + ", Loss: %.8e%n", total_loss / trainsetSize);

		} // end of all epochs

	}

	/**
	 * Calculate the cross entropy loss from the neural network for a single
	 * instance. The parameter is a single instance
	 */
	private double loss(Instance instance) {
		// TODO: add code here

		// calculates Cross-Entropy loss from the neural network for a single instance.
		// This function will be used by train()

		// in the backward pass, measure error for a single training example using
		// the Cross-Entropy loss function: CE = - sum(K) of (Ti * log(Oi))

		// K is the number of output units
		int K = this.outputNodes.size();
		int inputNodesSize = inputNodes.size();
		int hiddenNodesSize = hiddenNodes.size();

		double CE = 0.0;
		for (int i = 0; i < K; i++) {
			int T = instance.classValues.get(i);

			// do forward propagation before calculating loss
			//////////////////////////////////////////////////////
			// propagate the inputs forward to compute the outputs
			//////////////////////////////////////////////////////
			
			// for the input layer
			// set the input values for the input nodes
			for (int k = 0; k < inputNodesSize - 1; k++) { // excluding the bias node in the input layer
				// instance's attributes size is inputNodesSize - 1
				inputNodes.get(k).setInput(instance.attributes.get(k));
			}

			// for the hidden layer
			// calculate hidden nodes' output values
			for (int l = 0; l < hiddenNodesSize - 1; l++) { // excluding the bias node in the hidden layer
				hiddenNodes.get(l).setHiddenInput();
				hiddenNodes.get(l).calculateHiddenOutput();
			}

			// for the output layer
			double denominator = 0.0;
			// calculate output nodes' output values
			for (int o = 0; o < K; o++) {
				outputNodes.get(o).setOutputInput();
				// calculate the numerator of the Softmax equation
				outputNodes.get(o).calculateOutputNumerator();
				// accumulate the denominator of the Softmax equation
				denominator += Math.pow(Math.E, outputNodes.get(o).getInputValue());
			}

			for (int o = 0; o < K; o++) {
				outputNodes.get(o).calculateOutput_FinalSoftmaxOutput(denominator);
			}
			
			// this output values are from this particular instance
			double O = this.outputNodes.get(i).getOutput();
			if (O != 0.0) {
				double ln_O = Math.log(O);
				CE -= T * ln_O;
			}
		}
		return CE;
	}
}
