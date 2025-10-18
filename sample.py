import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-east-2")
inference_profile_arn = "arn:aws:bedrock:us-east-2:746630811346:inference-profile/us.amazon.nova-micro-v1:0"

body = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, Bedrock!"}
    ]
}

response = client.invoke_model(
    modelId=inference_profile_arn,
    body=json.dumps(body)
)

result = json.loads(response["body"].read())
print(result)


# {
#             "inferenceProfileName": "US Nova Micro",
#             "description": "Routes requests to Nova Micro in us-east-1, us-west-2 and us-east-2.",
#             "createdAt": "2024-11-29T13:23:00+00:00",
#             "updatedAt": "2025-10-02T05:11:58.324734+00:00",
#             "inferenceProfileArn": "arn:aws:bedrock:us-east-1:746630811346:inference-profile/us.amazon.nova-micro-v1:0",
#             "models": [
#                 {
#                     "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"
#                 },
#                 {
#                     "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-micro-v1:0"
#                 },
#                 {
#                     "modelArn": "arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-micro-v1:0"
#                 }
#             ],
#             "inferenceProfileId": "us.amazon.nova-micro-v1:0",
#             "status": "ACTIVE",
#              "type": "SYSTEM_DEFINED"
#         },