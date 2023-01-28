enum class DataType {
  kInvalidDataType = 0,
  kChar = 1,
  kFloat = 2,
  kDouble = 3,
  kInt8 = 4,
  kInt32 = 5,
  kInt64 = 6,
  kUInt8 = 7,
  kOFRecord = 8,
  kFloat16 = 9,
  kTensorBuffer = 10,
  kBFloat16 = 11,
  kBool = 12,
  kMaxDataType = 13
};

struct Params {
  DataType data_type;
  int64_t num_batches;
  int64_t num_heads;
  int64_t query_seq_len;
  int64_t kv_seq_len;
  int64_t head_size;
  int64_t value_head_size;
  int64_t query_hidden_stride;
  int64_t key_hidden_stride;
  int64_t value_hidden_stride;
  const void* causal_mask;
  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;
  void* out_ptr;
  void* workspace;
  int64_t workspace_size;
};