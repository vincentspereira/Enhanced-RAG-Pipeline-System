{{/*
Common labels
*/}}
{{- define "rag-system.labels" -}}
helm.sh/chart: {{ include "rag-system.chart" . }}
{{ include "rag-system.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "rag-system.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rag-system.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "rag-system.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fullname override for resources.
*/}}
{{- define "rag-system.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "rag-system.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
    {{ default (include "rag-system.fullname" .) .Values.serviceAccount.name }}
{{- else -}}
    {{ default "default" .Values.serviceAccount.name }}
{{- end -}}
{{- end -}}


{{/*
Helper to generate a ConfigMap name for a given component (e.g., internalApiGateway)
Usage: {{ include "rag-system.componentConfigMapName" (dict "componentName" "internalApiGateway" "context" $) }}
*/}}
{{- define "rag-system.componentConfigMapName" -}}
{{- printf "%s-%s-config" (include "rag-system.fullname" .context) .componentName | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Helper to generate a Deployment name for a given component
Usage: {{ include "rag-system.componentDeploymentName" (dict "componentName" "internalApiGateway" "context" $) }}
*/}}
{{- define "rag-system.componentDeploymentName" -}}
{{- printf "%s-%s" (include "rag-system.fullname" .context) .componentName | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Helper to generate a Service name for a given component
Usage: {{ include "rag-system.componentServiceName" (dict "componentName" "internalApiGateway" "context" $) }}
*/}}
{{- define "rag-system.componentServiceName" -}}
{{- printf "%s-%s" (include "rag-system.fullname" .context) .componentName | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common environment variables from a config block in values.yaml
Expects a dictionary from .Values.xxx.config
Example:
{{ include "rag-system.configEnvVars" ( dict "config" .Values.internalApiGateway.config ) }}
*/}}
{{- define "rag-system.configEnvVars" -}}
{{- range $key, $value := .config }}
- name: {{ $key | upper | replace "." "_" }}
  value: {{ $value | quote }}
{{- end }}
{{- end -}}
