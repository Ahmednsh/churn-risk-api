FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /source
COPY src/ChurnRisk.Api/ChurnRisk.Api.csproj src/ChurnRisk.Api/
COPY model/ model/
RUN dotnet restore src/ChurnRisk.Api/ChurnRisk.Api.csproj
COPY src/ChurnRisk.Api/ src/ChurnRisk.Api/
RUN dotnet publish src/ChurnRisk.Api/ChurnRisk.Api.csproj -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .
RUN mkdir /data && chown "$APP_UID" /data
ENV ChurnRisk__DbPath=/data/predictions.db
USER $APP_UID
EXPOSE 8080
ENTRYPOINT ["dotnet", "ChurnRisk.Api.dll"]
