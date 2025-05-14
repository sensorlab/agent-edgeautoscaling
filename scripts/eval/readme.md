# Evaluation Guide

This guide provides instructions for evaluating the MARLISE application using the provided scripts and configuration of LaaS (Localization as a Service).

## Prerequisites

- Kubernetes cluster with the LaaS microservices deployed.
- The microservices should be in pods, for example: `localization-api1`, `localization-api2`, etc.
- The backend should be running locally and available for scalable agents evaluation.

## Configuration

The evaluation is done with the separate [configuration](../../configs/localization/separate_services.yaml) of LaaS.

## Evaluation Scripts

### Dynamic Loading and Priority

The evaluation scripts for dynamic loading and priority work with multithreading. Ensure that your environment supports multithreading for optimal performance.

### Running the Scalable Agents evaluation

1. Ensure that the backend is running locally and accessible.
2. Run the evaluation script provided in this directory.

## Notes

- Make sure that the Kubernetes cluster is properly configured and the microservices are running in the specified pods.
- Adjust the configuration and evaluation scripts as needed to match your environment.
