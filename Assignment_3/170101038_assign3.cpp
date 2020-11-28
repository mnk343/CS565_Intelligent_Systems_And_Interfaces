/*
	CPP PROGRAM TO IMPLEMENT THE BIGRAM WORD2VEC MODEL
		Implementation details -> We are negative sampling in the assignment to speed up training,
								  more specifically, we will only update the weights connected to the output neurons

	AUTHOR: MAYANK WADHWANI
			170101038

	COMMAND TO RUN: g++ 170101038_assign3.cpp 
					./a.out input.txt output.txt

	Each line of output contains the following:
		<iteration_id> <pair_id> <count_of_negative_updates> <count_of_non_negative_updates>
*/

// Including all the necessary header files and using the standard namespace (std)
#include <iostream>
#include <vector>
#include <math.h>
#include <map>
using namespace std;

// Defining short declarations for long long and long double type
typedef long long int ll;
typedef long double ld;

// Function that implements the stochastic gradient descent algorithm
// Given a training example (input vector and output vector),
// we first feed forward in the neural network to get the predicted output
// and then backpropagate to make the updates
//
void feed_forward_and_backpropogate( vector<ll> &input_vector , vector<ll> &output_vector , vector< vector<ld> > &w_matrix , vector< vector<ld> > &w_prime_matrix , ll &number_of_negative_updates , ll &number_of_non_negative_updates , ld learning_rate){
	// FEED-FORWARD
	// We will first feed forward to predict the output

	// We first find the output at the hidden layer which will be the input at the final layer
	vector < ld > hidden_layer_output (w_matrix[0].size() , 0);
	for( ll j = 0 ;j < w_matrix[0].size() ; j ++ ){
		for( ll i = 0 ;i < w_matrix.size() ; i ++ ){
			hidden_layer_output[j] += w_matrix[i][j] * input_vector[i];
		}
	}
	
	// Using the output of the hidden layer and the W' matrix, we predict the output
	vector < ld > predicted_output (w_matrix.size() , 0);
	for( ll j = 0 ;j < w_prime_matrix[0].size() ; j ++ ){
		for( ll i = 0 ;i < w_prime_matrix.size() ; i ++ ){
			predicted_output[j] += w_prime_matrix[i][j] * hidden_layer_output[i];
		}
	}
	
	// BACK-PROPOGATION
	// We now turn to back-propogation, ie modify the W and W' matrices


	// We calculate the max value of an output neuron
	ld max_value = -1;
	for(ll i = 0; i < predicted_output.size() ; i ++){
		max_value = max(max_value , predicted_output[i]);
	}

	// To convert the outputs to the corresponding probabilities,
	// we use the softmax method
	// we also subtract the values from the max to avoid overshooting due
	// to the usage of exp function

	// We first find the sum of exp(output of neuron)
	ld total_sum = 0 ;
	
	for(ll i = 0; i < predicted_output.size() ; i ++){
		total_sum += (ld)exp(predicted_output[i]-max_value);
	}
	// Then find the softmax probabilities of each neuron
	vector< ld > softmax_probability ( w_matrix.size() , 0);
	for ( ll i = 0 ; i < w_matrix.size() ; i ++){
		softmax_probability[i] = (ld)(exp(predicted_output[i]-max_value) / total_sum);
	}

	// We then find the e (error e_j = h_j - t_j) values for each neuron
	vector < ld > e_value ( w_matrix.size() , 0);
	for ( ll i = 0 ; i < softmax_probability.size() ; i ++){
		e_value [i] = softmax_probability[i];
		// If output neuron is set, t_j is 1
		if( output_vector[i] == 1 ){
			e_value [i] -= 1;
		}
	}

	// We find the index where output neuron is set (1)
	ll index_output_neuron_set = 0;
	for( ll i = 0; i < output_vector.size() ; i ++){
		if( output_vector[i] == 1 ){
			index_output_neuron_set = i;
			break;
		}
	}

	// We now make a temporary copy of the W' matrix which will be later used to modify input matrix (W)
	vector < vector < ld > > w_prime_matrix_copy = w_prime_matrix;
	
	// We now make the necessary updates
	for( ll i = 0 ; i < w_matrix[0].size() ; i ++){
		
		// gradient = e_j * h_i where e is the error term for the jth neuron and h_i is the output of ith hidden neuron
		ld change_in_value = e_value[ index_output_neuron_set ] * hidden_layer_output[i];
		
		// Update the count of negative or non-negative updates appropriately
		if( change_in_value > 0 )
			number_of_negative_updates ++;
		else
			number_of_non_negative_updates ++;
		// Finally make the necessary changes to the W' matrix
		w_prime_matrix[i][index_output_neuron_set] -= learning_rate * change_in_value;
	}
	
	// We find the index where input neuron is set (1)
	ll index_input_neuron_set = 0;
	for( ll i = 0; i < input_vector.size() ; i ++){
		if( input_vector[i] == 1 ){
			index_input_neuron_set = i;
			break;
		}
	}

	// We now make the necessary updates
	for( ll i = 0 ; i < w_matrix[0].size() ; i ++){
		ld change_in_value = 0 ;

		// For each hidden layer, the change is the sum of e_j*w'_i_i which is also represented as EH in the research paper
		for( ll j = 0 ; j < w_matrix.size() ; j++ ){
			change_in_value += e_value[j] * w_prime_matrix_copy[i][j];
		}

		// Update the count of negative or non-negative updates appropriately
		if( change_in_value > 0 )
			number_of_negative_updates ++;
		else
			number_of_non_negative_updates ++;

		// We now make the necessary changes to the W matrix
		w_matrix[index_input_neuron_set][i] -= learning_rate * change_in_value ;
	}
}

// Function to perform a single iteration on the complete training set
void perform_single_iteration( ll iteration_id , vector<pair< ll , ll> > &training_set , vector<vector< ld > > &w_matrix , vector<vector< ld > > &w_prime_matrix , ld learning_rate ){

	// Iterate over the training set
	for( ll i = 0 ; i < training_set.size(); i ++){
		ll number_of_negative_updates = 0;
		ll number_of_non_negative_updates = 0;

		// Create the input vector and the output vector based on the particular training example 
		vector<ll> input_vector( w_matrix.size() , 0 );
		input_vector [ training_set[i].first - 1 ] = 1;

		vector<ll> output_vector( w_matrix.size() , 0 );
		output_vector [ training_set[i].second - 1 ] = 1;

		// Call the function to update the weights and find the required counts of negative and non-negative updates
		feed_forward_and_backpropogate( input_vector , output_vector , w_matrix , w_prime_matrix , number_of_negative_updates , number_of_non_negative_updates , learning_rate );

		// Print everything in file
		cout << iteration_id << " " << i+1 << " " << number_of_negative_updates << " " << number_of_non_negative_updates << "\n";
	}
}

// The main function to implement the bigram word2vec model
int main(int argc , char ** argv){

    // We take all the necessary inputs
    // Input file format is as follows:
    // Line 1: Count of words in the vocabulary
    // Line 2: Count of hidden layer neurons / dimensionality of word vectors
    // Line 3: Learning rate
    // Line 4: Number of iterations to perform
    // Line 5: Number of word pairs to be given as input (n)
    // Line 6-6+n-1: <Input Word Id> <Output Word ID> 

    ll count_of_distinct_words_vocabulary;
    cin >> count_of_distinct_words_vocabulary;

    ll count_of_hidden_layer_neurons;
    cin >> count_of_hidden_layer_neurons;

    ld learning_rate;
    cin >> learning_rate;

    ll number_of_iterations;
    cin >> number_of_iterations;

    ll count_word_pairs;
    cin>> count_word_pairs;

    vector < pair<ll,ll> > training_set;
    for( ll i = 0 ;i < count_word_pairs; i ++){
    	ll pair_id;
    	cin>>pair_id;

    	ll x,y;
    	cin >> x >> y;
    	training_set.push_back( make_pair(x,y) );
    }

    // We generate the W and the W' matrices which correponds to matrices for
    // input to hidden layer and hidden to output layer
    vector< vector< ld > > w_matrix( count_of_distinct_words_vocabulary , vector< ld > ( count_of_hidden_layer_neurons , 0.5 ) );
    vector< vector< ld > > w_prime_matrix( count_of_hidden_layer_neurons , vector< ld > (count_of_distinct_words_vocabulary , 0.5 ) );

    // We now perform iterations on the training set
    for( ll i = 1 ; i <= number_of_iterations ; i ++){
    	perform_single_iteration( i , training_set , w_matrix , w_prime_matrix , learning_rate);
    }

	return 0;
}

// END OF PROGRAM