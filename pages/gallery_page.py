import streamlit as st
import io
from PIL import Image

st.set_page_config(layout="wide", page_title="Galeria de Gráficos ENEM")

try:
    # Assuming main.py might have a function to restore filters to session_state
    # For the gallery, this is primarily used by the "Aplicar Filtros e Voltar" button
    # which sets st.session_state.restore_filters_data
    pass  # No explicit restore_filters function is called directly at the top of gallery.py
except ImportError:
    st.error(
        "Erro: Problema ao verificar importações de main.py. Certifique-se de que 'main.py' existe e está no diretório correto se funcionalidades dependentes forem necessárias."
    )
    # No st.stop() here unless main.py is absolutely critical for gallery's existence

st.title("🖼️ Sua Galeria de Gráficos Salvos")
st.markdown(
    """
    Aqui você pode visualizar todos os gráficos que foram salvos durante sua análise.
    Use as opções de busca, ordenação e exclusão para gerenciar sua galeria.
    Clique em um gráfico para ver os filtros exatos usados para gerá-lo e, opcionalmente,
    retornar à página principal com esses filtros aplicados.
    """
)

if "gallery_items" not in st.session_state or not st.session_state.gallery_items:
    st.info(
        "Nenhum gráfico foi salvo ainda. Volte para a página principal para salvar alguns!"
    )
    st.image("https://i.imgur.com/gK7e9s7.png")  # Placeholder image
else:
    # --- Preparação dos Dados para Exibição da Galeria ---
    # Garantir que todos os itens tenham as chaves necessárias.
    for item in st.session_state.gallery_items:
        item.setdefault("plot_type", "Desconhecido")
        item.setdefault("chart_axes", "N/A")  # Still useful for display
        # Add defaults for new filter keys if they don't exist (for backward compatibility)
        item.setdefault("c_type", "N/A")
        item.setdefault("selected_classification", "N/A")
        item.setdefault("feature_name", "N/A")
        item.setdefault("title", "Título Desconhecido")  # Ensure title always exists
        item.setdefault("filters", {})  # Ensure filters dict always exists
        item.setdefault("image_bytes", None)  # Ensure image_bytes always exists

    # Sidebar para controles
    st.sidebar.header("Opções da Galeria")

    # Funcionalidade de Busca por Título
    search_query_title = st.sidebar.text_input("Buscar por título:", "").lower()

    # --- NOVO: Selects Individuais para Componentes do Eixo ---
    st.sidebar.subheader("Filtrar por Componentes do Eixo:")

    # Get unique values for select boxes, adding a "Todos" option
    all_c_types = sorted(
        list(
            set(
                ["Todos"]
                + [
                    item["c_type"]
                    for item in st.session_state.gallery_items
                    if item.get("c_type")
                ]
            )
        )
    )
    selected_c_type_filter = st.sidebar.selectbox(
        "Tipo Classificação:", all_c_types, key="select_c_type"
    )

    all_categories = sorted(
        list(
            set(
                ["Todas"]
                + [
                    item["selected_classification"]
                    for item in st.session_state.gallery_items
                    if item.get("selected_classification")
                ]
            )
        )
    )
    selected_category_filter = st.sidebar.selectbox(
        "Categoria:", all_categories, key="select_category"
    )

    all_features = sorted(
        list(
            set(
                ["Todas"]
                + [
                    item["feature_name"]
                    for item in st.session_state.gallery_items
                    if item.get("feature_name")
                ]
            )
        )
    )
    selected_feature_filter = st.sidebar.selectbox(
        "Feature:", all_features, key="select_feature"
    )

    # Funcionalidade de Ordenação
    sort_options = {
        "Mais Recente": "most_recent",
        "Mais Antigo": "oldest",
        "Ordem Alfabética (A-Z)": "alpha_asc",
        "Ordem Alfabética (Z-A)": "alpha_desc",
    }
    selected_sort = st.sidebar.selectbox("Ordenar por:", list(sort_options.keys()))

    # Filtrar por tipo de gráfico
    all_plot_types = sorted(
        list(set([item["plot_type"] for item in st.session_state.gallery_items]))
    )
    filter_plot_type = st.sidebar.multiselect(
        "Filtrar por Tipo de Gráfico:", all_plot_types, default=all_plot_types
    )

    # Aplicar busca e filtros
    filtered_items = [
        item
        for item in st.session_state.gallery_items
        if search_query_title in item["title"].lower()
    ]

    # Aplicar filtro de Tipo Classificação
    if selected_c_type_filter != "Todos":
        filtered_items = [
            item for item in filtered_items if item["c_type"] == selected_c_type_filter
        ]

    # Aplicar filtro de Categoria
    if selected_category_filter != "Todas":
        filtered_items = [
            item
            for item in filtered_items
            if item["selected_classification"] == selected_category_filter
        ]

    # Aplicar filtro de Feature
    if selected_feature_filter != "Todas":
        filtered_items = [
            item
            for item in filtered_items
            if item["feature_name"] == selected_feature_filter
        ]

    # Aplicar filtro de tipo de gráfico
    filtered_items = [
        item for item in filtered_items if item["plot_type"] in filter_plot_type
    ]

    # Aplicar ordenação
    if sort_options[selected_sort] == "most_recent":
        # Assuming gallery_items are added chronologically, reverse will put newest first
        # If there's a timestamp, sort by that. Otherwise, current order is oldest first.
        display_items = filtered_items[::-1]
    elif sort_options[selected_sort] == "oldest":
        display_items = filtered_items  # Original order (appended)
    elif sort_options[selected_sort] == "alpha_asc":
        display_items = sorted(filtered_items, key=lambda x: x["title"])
    elif sort_options[selected_sort] == "alpha_desc":
        display_items = sorted(filtered_items, key=lambda x: x["title"], reverse=True)
    else:
        display_items = filtered_items  # Default

    if not display_items:
        st.info("Nenhum gráfico corresponde aos seus critérios de busca/filtro.")
    else:
        for i, item in enumerate(display_items):
            st.markdown(
                f"### {item.get('title', 'Gráfico Sem Título')}"
            )  # Use .get for safety
            col1, col2 = st.columns([3, 1])

            with col1:
                if item.get("image_bytes"):
                    try:
                        img = Image.open(io.BytesIO(item["image_bytes"]))
                        st.image(
                            img, use_container_width=True
                        )  # Changed to True for better scaling
                    except Exception as e:
                        st.error(f"Erro ao carregar a imagem: {e}")
                else:
                    st.warning("Dados da imagem não encontrados para este gráfico.")

            with col2:
                st.subheader("Filtros Utilizados:")
                # Exibir os filtros de forma mais legível
                if item.get("filters"):
                    for filter_name, filter_value in item["filters"].items():
                        display_name = filter_name.replace("_", " ").title()
                        if isinstance(filter_value, list):
                            st.write(
                                f"**{display_name}:** {', '.join(map(str, filter_value))}"
                            )
                        else:
                            st.write(f"**{display_name}:** {filter_value}")
                else:
                    st.write("Nenhum filtro registrado para este gráfico.")

                # Exibir Eixos do Gráfico (original string)
                st.write(
                    f"**Eixos do Gráfico (Original):** {item.get('chart_axes', 'N/A')}"
                )
                # Exibir componentes individuais se disponíveis
                st.write(f"**Tipo Classificação:** {item.get('c_type', 'N/A')}")
                st.write(f"**Categoria:** {item.get('selected_classification', 'N/A')}")
                st.write(f"**Feature:** {item.get('feature_name', 'N/A')}")

                if st.button(f"Aplicar Filtros e Voltar", key=f"restore_btn_{i}"):
                    st.session_state.restore_filters_data = item["filters"]
                    # Ensure main.py exists if you use st.switch_page
                    try:
                        st.switch_page("main.py")
                    except st.errors.StreamlitAPIException as e:
                        if "main.py has not been found" in str(e):
                            st.error(
                                "Erro: A página 'main.py' não foi encontrada. Não é possível retornar."
                            )
                        else:
                            raise e

                if st.button(
                    f"Deletar Gráfico",
                    key=f"delete_btn_{i}",
                    help="Remover este gráfico da galeria",
                ):
                    # To correctly identify the item to delete from the original list,
                    # we need a reliable unique identifier. Using the title and potentially
                    # other specific attributes stored at save time.
                    # For robustness, it's better to find the item by a unique ID if one was assigned
                    # or by matching multiple key attributes.
                    # The original deletion logic was based on a tuple of attributes.
                    # This needs to be robust to changes in how items are structured.
                    # A simpler way if titles are unique (or unique enough in practice):
                    # Or, iterate and match the specific 'item' object if it's from the session_state list directly
                    # However, 'display_items' is a filtered/sorted copy.
                    # We need to find the corresponding item in the original st.session_state.gallery_items

                    item_to_delete_original_index = None
                    for original_idx, original_item in enumerate(
                        st.session_state.gallery_items
                    ):
                        # Create an ID for the currently displayed item and the original item
                        # This assumes these fields are consistently present and define uniqueness
                        current_item_id_tuple = (
                            item.get("title"),
                            item.get("plot_type"),
                            item.get(
                                "chart_axes"
                            ),  # Keep for ID if it's still a defining characteristic
                            item.get("c_type"),
                            item.get("selected_classification"),
                            item.get("feature_name"),
                            # Potentially add more fields if needed for uniqueness, like image_bytes hash or a timestamp
                        )
                        original_item_id_tuple = (
                            original_item.get("title"),
                            original_item.get("plot_type"),
                            original_item.get("chart_axes"),
                            original_item.get("c_type"),
                            original_item.get("selected_classification"),
                            original_item.get("feature_name"),
                        )
                        if current_item_id_tuple == original_item_id_tuple:
                            # More robust check: compare image_bytes if available, as titles might not be unique
                            if item.get("image_bytes") == original_item.get(
                                "image_bytes"
                            ):
                                item_to_delete_original_index = original_idx
                                break

                    if item_to_delete_original_index is not None:
                        deleted_item_title = st.session_state.gallery_items[
                            item_to_delete_original_index
                        ].get("title", "Gráfico")
                        del st.session_state.gallery_items[
                            item_to_delete_original_index
                        ]
                        st.success(f"Gráfico '{deleted_item_title}' deletado!")
                        st.rerun()
                    else:
                        st.error(
                            "Não foi possível encontrar o gráfico para deletar. Tente recarregar."
                        )

            st.markdown("---")

# Opção para limpar todos os gráficos
st.sidebar.markdown("---")
if st.sidebar.button(
    "Limpar Todos os Gráficos",
    help="Remove todos os gráficos da galeria",
    key="clear_all_initial_btn",
):
    # Using a flag in session_state to manage confirmation visibility
    st.session_state.confirm_clear_all_visible = True

if st.session_state.get("confirm_clear_all_visible", False):
    st.sidebar.warning(
        "Tem certeza que deseja apagar **TODOS** os gráficos da galeria?"
    )
    col_confirm, col_cancel = st.sidebar.columns(2)
    if col_confirm.button("Confirmar Limpar Todos", key="confirm_clear_all_btn_final"):
        st.session_state.gallery_items = []
        st.session_state.confirm_clear_all_visible = False  # Hide confirmation
        st.success("Todos os gráficos foram removidos da galeria.")
        st.rerun()
    if col_cancel.button("Cancelar", key="cancel_clear_all_btn"):
        st.session_state.confirm_clear_all_visible = False  # Hide confirmation
        st.rerun()


st.sidebar.markdown("---")
if st.sidebar.button("Voltar para Análise Principal", key="back_to_main_btn"):
    try:
        st.switch_page("main.py")
    except st.errors.StreamlitAPIException as e:
        if "main.py has not been found" in str(e):
            st.error("Erro: A página 'main.py' não foi encontrada.")
        else:
            raise e
