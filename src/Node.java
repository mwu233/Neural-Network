///////////////////////////////////////////////////////////////////////////////
// Project:			 Homework 4	
// Main Class File:  DigitClassifier.java
// File:             Node.java
// Semester:         CS540 Section 2 Spring 2018
//
// Author:           Meiliu Wu (mwu233@wisc.edu)
// Lecturer's Name:  Chuck Dyer
///////////////////////////////////////////////////////////////////////////////

import java.util.*;

/**
 * Class for internal organization of a Neural Network. There are 5 types of
 * nodes. Check the type attribute of the node for details. Feel free to modify
 * the provided function signatures to fit your own implementation
 * 
 * <p>Bugs: Not found.
 * 
 * @author Meiliu Wu
 */

public class Node {
	private int type = 0; // 0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents = null; // Array List that will contain the parents (including the bias
														// node) with weights if applicable

	private double inputValue = 0.0;
	private double outputValue = 0.0;
	private double outputGradient = 0.0;
	private double delta = 0.0; // input gradient

	// Create a node with a specific type
	Node(int type) {
		if (type > 4 || type < 0) {
			System.out.println("Incorrect value for node type");
			System.exit(1);

		} else {
			this.type = type;
		}

		if (type == 2 || type == 4) {
			parents = new ArrayList<>();
		}
	}

	// For an input node sets the input value which will be the value of a
	// particular attribute
	public void setInput(double inputValue) {
		if (type == 0) { // If input node
			this.inputValue = inputValue;
		}
	}

	// For an hidden node sets the input value
	public void setHiddenInput() {
		this.inputValue = 0.0;
		// setup the input value (weighted sum) for the node
		for (int i = 0; i < this.parents.size(); i++) {
			this.inputValue += this.parents.get(i).node.getOutput() * this.parents.get(i).weight;
		}
	}

	// For an output node sets the input value
	public void setOutputInput() {
		this.inputValue = 0.0;
		// setup the input value (weighted sum) for the node
		for (int i = 0; i < this.parents.size(); i++) {
			this.inputValue += this.parents.get(i).node.getOutput() * this.parents.get(i).weight;
		}
	}

	public double getInputValue() {
		return this.inputValue;
	}

	/**
	 * Calculate the output of a node. You can get this value by using getOutput()
	 */
	public void calculateOutput() {
		if (type == 2 || type == 4) { // Not an input or bias node
			// TODO: add code here
		}
	}

	public void calculateHiddenOutput() {
		if (this.inputValue > 0) {
			this.outputValue = this.inputValue;
		} else {
			this.outputValue = 0.0;
		}
	}

	public void calculateOutputNumerator() {
		// the outputValue for now is the numerator of the Softmax equation
		this.outputValue = Math.pow(Math.E, this.inputValue);
	}

	public void calculateOutput_FinalSoftmaxOutput(double denominator) {
		this.outputValue /= denominator;

	}

	// Gets the output value
	public double getOutput() {

		if (type == 0) { // Input node
			return inputValue;
		} else if (type == 1 || type == 3) { // Bias node
			return 1.00;
		} else {
			return outputValue;
		}

	}

	// Calculate the delta value of a node.
	// delta value is the input gradient
	public void calculateDelta() {
		if (type == 2 || type == 4) {
			// TODO: add code here
		}
	}

	public void calculateOutput_Target0_Delta() {
		this.delta = 0.0 - this.getOutput();
	}

	public void calculateOutput_Target1_Delta() {
		this.delta = 1.0 - this.getOutput();
	}

	public void calculateHidden_OutputGradient() {
		// only calculate the first part of the delta
		// i.e., derivative of RELU(weighted sum input)
		if (this.inputValue > 0) {
			this.outputGradient = 1.0;
		} else {
			this.outputGradient = 0.0;
		}
	}

	public void calculateHidden_Delta(double sum) {
		this.delta = this.outputGradient * sum;
	}

	public double getDelta() {
		return delta;
	}

	// Update the weights between parents node and current node
	public void updateWeight(double learningRate) {
		if (type == 2 || type == 4) {
			// TODO: add code here
			// current node is the node on the layer at the right side
			for (int i = 0; i < this.parents.size(); i++) {
				this.parents.get(i).weight += learningRate * this.parents.get(i).node.getOutput() * this.getDelta();
			}
		}
	}
}
