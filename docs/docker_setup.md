# Docker Setup

The repository includes a simple Docker build and Compose stack for the FastAPI service.

## Build and run

```bash
docker compose up --build
```

The Compose file passes `NVIDIA_API_KEY` into the container at runtime and does not bake secrets into the image.

## Service ports

- Host: `8000`
- Container: `8000`

## Mounted paths

- `./audits` -> `/app/audits`
- `./docs` -> `/app/docs`

## Notes

- Keep `.env` local and ignored by git.
- Use `docker compose` with a populated local `.env` file or exported environment variable for `NVIDIA_API_KEY`.
