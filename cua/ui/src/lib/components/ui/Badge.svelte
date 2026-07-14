<script lang="ts">
  export let tone: "default" | "muted" | "bad" = "default";
</script>

<span class="ui-badge {tone}"><slot /></span>
