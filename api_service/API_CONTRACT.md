# AI Service API Contract

## Endpoint

- Method: `POST`
- Path: `/api/v1/extract-faces`
- Content-Type: `application/json`

## Request JSON

```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBx...",
  "camera_id": "door_01_cam",
  "timestamp": "2026-04-24T08:15:30.123Z",
  "require_alignment": true
}
```

### Field Notes

- `image_base64`: Required. Base64 encoded image bytes.
- `camera_id`: Optional. Helpful for logging/tracing camera source.
- `timestamp`: Optional. Frame capture timestamp passed through for tracing.
- `require_alignment`: Optional. If `false`, faces are detected and boxed, but embeddings are skipped (`embedding` becomes an empty array).

## Success Response

- Code: `200 OK`

```json
{
  "status": "success",
  "faces_count": 2,
  "processing_time_ms": 142,
  "data": [
    {
      "face_index": 0,
      "confidence": 0.98,
      "bounding_box": {
        "x": 150,
        "y": 80,
        "width": 200,
        "height": 200
      },
      "embedding": [0.142, -0.088, 0.991, -0.341]
    },
    {
      "face_index": 1,
      "confidence": 0.91,
      "bounding_box": {
        "x": 500,
        "y": 120,
        "width": 180,
        "height": 195
      },
      "embedding": [-0.551, 0.112, 0.045, 0.876]
    }
  ]
}
```

## Error Response

- Code: `400 Bad Request` or `500 Internal Server Error`

```json
{
  "status": "error",
  "error_code": "IMAGE_UNREADABLE",
  "message": "The provided base64 string could not be decoded into a valid image matrix.",
  "processing_time_ms": 12
}
```

## Typical Error Codes

- `INVALID_JSON`
- `MISSING_IMAGE`
- `IMAGE_UNREADABLE`
- `INTERNAL_ERROR`
