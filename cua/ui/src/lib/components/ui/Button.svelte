<script lang="ts">
  export let variant: "default" | "secondary" | "ghost" = "default";
  export let disabled = false;
  let className = "";
  export { className as class };
</script>

<button class="ui-button {variant} {className}" {disabled} on:click>
  <slot />
</button>
