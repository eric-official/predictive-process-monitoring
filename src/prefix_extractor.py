import pandas as pd


class PrefixExtractor:

    def __init__(self, bucketing_lower_bound=2, bucketing_upper_bound=20, add_age=False, add_temporal_information=False,
                 encoding_technique="Aggregation", num_additional_attributes=0, last_state_offset=0, add_age_offset=0):
        """
        PrefixExtractor handles to step of extracting prefixes from the event streams
        @param bucketing_lower_bound: minimum length for prefix length (must be equal bucketing_upper_bound for
                                      bucketing techniques Clustering and SingleBucket)
        @param bucketing_upper_bound: maximum length for prefix length (must be equal bucketing_lower_bound for
                                      bucketing techniques Clustering and SingleBucket)
        @param add_age: boolean indicating inclusion of age attribute
        @param add_temporal_information: boolean indicating integration of month, weekday and hour
        @param encoding_technique: name of encoding technique ("Aggregation", "LastState")
        @param num_additional_attributes: number of additional attributes (LastState, add_age, add_temporal_information)
        @param last_state_offset: number of attributes from last column to last state attribute
        @param add_age_offset: number of attributes from last column to age attribute
        """

        self.bucketing_lower_bound = bucketing_lower_bound
        self.bucketing_upper_bound = bucketing_upper_bound
        self.add_age = add_age
        self.add_temporal_information = add_temporal_information
        self.encoding_technique = encoding_technique
        self.num_additional_attributes = num_additional_attributes
        self.last_state_offset = last_state_offset
        self.add_age_offset = add_age_offset
        self.prefixes_train = None
        self.prefixes_val = None
        self.prefixes_test = None

    def extract_prefixes(self, stream_train, stream_val, stream_test):
        """
        creates prefixes of with defined lengths from list of streams
        @param stream_train: event streams from training data
        @param stream_val: event streams from validation data
        @param stream_test: event streams from testing data
        """

        list_of_prefix_dicts = []

        # iterate over all collections of event streams
        for stream in [stream_train, stream_val, stream_test]:

            # create items for each defined prefix length
            prefix_dict = {key: [] for key in range(self.bucketing_lower_bound, self.bucketing_upper_bound + 1)}

            # iterate over all traces in stream collection
            for trace in stream:

                # add prefixes for each defined prefix length
                for prefix_length in range(self.bucketing_lower_bound, self.bucketing_upper_bound + 1):

                    # extract prefix of certain length
                    trace_length_copy = trace[:prefix_length]
                    extracted_prefix = []

                    # to create final prefix, take only categorical values (activities) first
                    for element in trace_length_copy:
                        if isinstance(element, str):
                            extracted_prefix.append(element)

                    # now fill up extracted prefix with None to certain length if necessary
                    while len(extracted_prefix) < prefix_length:
                        extracted_prefix.append(None)

                    # add last state if defined
                    if self.encoding_technique == "LastState":
                        extracted_prefix.append(trace[len(trace) - 1 - self.last_state_offset])

                    # add age if defined
                    if self.add_age is True:
                        extracted_prefix.append(trace[len(trace) - 1 - self.add_age_offset])

                    # add temporal information if necessary
                    if self.add_temporal_information is True and all(
                            extracted_prefix[len(extracted_prefix) - i] != trace[len(trace) - i] for i in range(2, 5)):
                        extracted_prefix.append(trace[len(trace) - 4])
                        extracted_prefix.append(trace[len(trace) - 3])
                        extracted_prefix.append(trace[len(trace) - 2])

                    # add remaining time to extracted prefix
                    if extracted_prefix[len(extracted_prefix) - 1] != trace[len(trace) - 1]:
                        extracted_prefix.append(trace[len(trace) - 1])

                    prefix_dict[prefix_length].append(extracted_prefix)

            list_of_prefix_dicts.append(prefix_dict)

        self.prefixes_train = list_of_prefix_dicts[0]
        self.prefixes_val = list_of_prefix_dicts[1]
        self.prefixes_test = list_of_prefix_dicts[2]

    def transform_prefix_to_df(self):
        """
        takes each list value (list of lists) in the dictionary and stores the created dataframe at the corresponding
        key
        """

        for key, value in self.prefixes_train.items():
            self.prefixes_train[key] = pd.DataFrame(value)

        for key, value in self.prefixes_val.items():
            self.prefixes_val[key] = pd.DataFrame(value)

        for key, value in self.prefixes_test.items():
            self.prefixes_test[key] = pd.DataFrame(value)
