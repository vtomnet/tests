<script lang="ts">
  export let id = "";
  export let value = "";
  export let placeholder = "";
  let className = "";
  export { className as class };
</script>

<input {id} class="ui-input {className}" bind:value {placeholder} on:input on:change />
