"""
URL configuration for arxrag project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rag.views import ask
from rag.agent import (agent_search_ingest, agent_ask, agent_list_documents, agent_get_document,
    agent_list_chunks, agent_get_chunk, agent_list_references, agent_get_reference,
    agent_search_references, agent_first_reference, agent_search_chunks, agent_tool_catalog, agent_agentic_ask)
from rag.views import home

urlpatterns = [
    path("", home),
    path("admin/", admin.site.urls),
    path("api/ask", ask),
    path("api/agent/search_ingest", agent_search_ingest),
    path("api/agent/ask", agent_ask),
    path("api/agent/agentic_ask", agent_agentic_ask),
    path("api/agent/documents", agent_list_documents),
    path("api/agent/tools", agent_tool_catalog),
    path("api/agent/documents/<int:doc_id>", agent_get_document),
    path("api/agent/documents/<int:doc_id>/chunks", agent_list_chunks),
    path("api/agent/documents/<int:doc_id>/chunks/<int:ord>", agent_get_chunk),
    path("api/agent/documents/<int:doc_id>/references", agent_list_references),
    path("api/agent/documents/<int:doc_id>/references/<int:position>", agent_get_reference),
    path("api/agent/documents/<int:doc_id>/first_reference", agent_first_reference),
    path("api/agent/search_references", agent_search_references),
    path("api/agent/search_chunks", agent_search_chunks),
]


