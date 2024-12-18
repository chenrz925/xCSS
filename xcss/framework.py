from dependency_injector import containers, providers


class XCSS(containers.DeclarativeContainer):
    config = providers.Configuration()
